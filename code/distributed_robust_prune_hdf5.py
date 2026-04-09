import numpy as np
import collections
import time
from heapq import heappush
import ray
import os
import h5py

os.environ["RAY_DEDUP_LOGS"] = "0"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5",            required=True,  help="Path to HDF5 dataset file")
    parser.add_argument("--hdf5_key",        default="train", help="Key inside the HDF5 file (default: train)")
    parser.add_argument("--name",            required=True,  help="Dataset name used for output file naming")
    parser.add_argument("--save_path",       required=True,  help="Directory to save results")
    parser.add_argument("--num_points",      type=int, default=0,
                        help="Number of neighborhoods to compute (0 = all points in dataset)")
    parser.add_argument("--batch",           type=int, default=50)
    parser.add_argument("--num_shards",      type=int, default=17)
    parser.add_argument("--cpus",            type=int, default=16)
    parser.add_argument("--coordinator_cpus", type=int, default=1)
    args = parser.parse_args()

    # Resolve num_points: 0 means use the full dataset
    if args.num_points == 0:
        with h5py.File(args.hdf5, 'r') as f:
            args.num_points = f[args.hdf5_key].shape[0]
        print(f"num_points set to full dataset size: {args.num_points:,}", flush=True)

    ray.init(address="auto", runtime_env={
        "env_vars": {
            "OMP_NUM_THREADS":      str(args.cpus),
            "OPENBLAS_NUM_THREADS": str(args.cpus),
        }
    })

    expected_cpus = args.num_shards * args.cpus + args.coordinator_cpus
    print(f"Waiting for {args.num_shards} workers ({expected_cpus} CPUs)...", flush=True)
    while True:
        available = ray.cluster_resources().get("CPU", 0)
        print(f"  CPUs available: {available:.0f}/{expected_cpus}", flush=True)
        if available >= expected_cpus:
            break
        time.sleep(15)
    print("All workers ready.", flush=True)

    print(f"Creating {args.num_shards} worker actors...", flush=True)
    workers = [
        WorkerActor.options(num_cpus=args.cpus).remote(
            i, args.hdf5, args.hdf5_key, args.num_shards, args.cpus
        )
        for i in range(args.num_shards)
    ]

    pending   = {w.ready.remote(): i for i, w in enumerate(workers)}
    remaining = set(pending.keys())
    while remaining:
        done, remaining = ray.wait(list(remaining), num_returns=1, timeout=30)
        for fut in done:
            print(f"  Worker {pending[fut]} ready.", flush=True)
        if remaining:
            still_waiting = sorted(pending[f] for f in remaining)
            print(f"  Still waiting on workers: {still_waiting}", flush=True)
    print("All workers initialized.", flush=True)

    print("Creating coordinator actor...", flush=True)
    coordinator = CoordinatorActor.options(num_cpus=args.coordinator_cpus).remote(
        hdf5_path=args.hdf5,
        hdf5_key=args.hdf5_key,
        name=args.name,
        save_path=args.save_path,
        num_points=args.num_points,
        batch=args.batch,
        workers=workers,
    )
    print("Waiting for coordinator to initialize...", flush=True)
    ray.get(coordinator.ready.remote())
    print("Coordinator ready. Starting computation.", flush=True)

    ray.get(coordinator.computeNeighborhoods.remote())


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    def __init__(self, hdf5_path, hdf5_key, name, save_path, num_points, batch):
        os.environ["RAY_DEDUP_LOGS"] = "0"
        self.name       = name
        self.save_path  = save_path
        self.num_points = num_points
        self.batch      = batch

        # Keep the HDF5 file open for lazy row access.
        # h5py fancy indexing requires sorted indices; use _fetch_vecs() for this.
        self._h5_file   = h5py.File(hdf5_path, 'r')
        self.h5_dataset = self._h5_file[hdf5_key]
        total           = self.h5_dataset.shape[0]

        # IDs are simply 0..total-1
        self.vector_ids = np.arange(total, dtype=np.int64)

        self.neighborhoods = {}
        self.active        = set()
        self.computed      = set()
        self.start_times   = {}
        self.round_num     = 0
        self.rtt_history   = []
        self.uncov_initial = {}
        self.uncov_current = {}
        self.point_state   = {}
        self.degrees = []

        computed_path = self._computed_path()
        if os.path.exists(computed_path):
            with open(computed_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.computed.add(int(line))

        print(f"Resuming from {len(self.computed)} already computed neighborhoods.", flush=True)

    def _computed_path(self):
        return os.path.join(self.save_path, f"{self.name}-computed.txt")

    def _adj_path(self):
        return os.path.join(self.save_path, f"adj-list-{self.name}-euclidean.txt")

    def _fetch_vecs(self, indices):
        """Fetch rows from the HDF5 dataset.
        h5py fancy indexing requires strictly increasing indices with no duplicates.
        np.unique returns sorted unique values; inverse reconstructs original order."""
        idx = np.asarray(indices, dtype=np.int64)
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        vecs_unique = self.h5_dataset[unique_idx.tolist()].astype(np.float32)
        return vecs_unique[inverse]

    def ready(self):
        return True

    def computeNeighborhoods(self):
        print("[Coordinator] computeNeighborhoods start", flush=True)
        initial_sample = np.random.choice(self.vector_ids, self.batch, replace=False)
        self.active    = set(int(v) for v in initial_sample)
        print(f"[Coordinator] initial batch sampled: {self.active}", flush=True)

        message           = []
        compute_distances = []
        for vec_id in self.active:
            message.append((vec_id, 'INIT', None))
            compute_distances.append(vec_id)
            self.neighborhoods[vec_id] = []
            self.start_times[vec_id]   = time.time()
            self.point_state[vec_id]   = 'INIT'

        responses, rtt, uncov_totals = self.sendMessages(compute_distances, message)
        for vec_id, count in uncov_totals.items():
            self.uncov_initial[vec_id] = count
            self.uncov_current[vec_id] = count
        print(f"[Round 0] INIT  rtt={rtt:.2f}s  active={len(self.active)}", flush=True)

        while self.active:
            compute_distances = []
            message           = []

            for vec_id in list(self.active):
                if vec_id not in responses or responses[vec_id] is None:
                    message.append((vec_id, 'KILL', None))
                    self.degrees.append(len(self.neighborhoods[vec_id]))
                    self.writeNeighborhood(vec_id)
                    self.active.remove(vec_id)
                    self.computed.add(vec_id)
                    self.uncov_initial.pop(vec_id, None)
                    self.uncov_current.pop(vec_id, None)
                    self.point_state.pop(vec_id, None)

                    if (len(self.active) + len(self.computed)) < self.num_points:
                        new_vec = int(np.random.choice(self.vector_ids))
                        while new_vec in self.computed:
                            new_vec = int(np.random.choice(self.vector_ids))
                        self.active.add(new_vec)
                        message.append((new_vec, 'INIT', None))
                        compute_distances.append(new_vec)
                        self.neighborhoods[new_vec] = []
                        self.start_times[new_vec]   = time.time()
                        self.point_state[new_vec]   = 'INIT'
                else:
                    _, neighbor = responses[vec_id]
                    self.neighborhoods[vec_id].append((neighbor, self.uncov_current[vec_id]))
                    message.append((vec_id, 'UPDATE', neighbor))
                    compute_distances.append(neighbor)
                    self.point_state[vec_id] = 'UPDATE'

            responses, rtt, uncov_totals = self.sendMessages(compute_distances, message)
            self.round_num += 1
            self.rtt_history.append(rtt)

            for vec_id, count in uncov_totals.items():
                if vec_id not in self.uncov_initial:
                    self.uncov_initial[vec_id] = count
                self.uncov_current[vec_id] = count

            self._print_round()

        print(f"\nAll {len(self.computed)} neighborhoods computed.", flush=True)

    def _print_round(self):
        rtt_min = min(self.rtt_history)
        rtt_avg = sum(self.rtt_history) / len(self.rtt_history)
        rtt_max = max(self.rtt_history)

        avg_deg = np.mean(self.degrees) if self.degrees else np.inf

        lines = [
            f"\n[Round {self.round_num}] Took {self.rtt_history[-1]:.1f}s |\n"
            f"RTT min={rtt_min:.1f}s avg={rtt_avg:.1f}s max={rtt_max:.1f}s  "
            f"active={len(self.active)}  completed={len(self.computed)}\n"
            f"goal ETA={(self.num_points - len(self.computed)) * rtt_avg * avg_deg / self.batch:.2f}s "
            f"(assuming avg deg = {avg_deg})"
        ]

        now = time.time()
        for vec_id in sorted(self.active):
            uncov    = self.uncov_current.get(vec_id, 0)
            initial  = self.uncov_initial.get(vec_id, uncov) or 1
            covered  = initial - uncov
            progress = max(0.0, covered / initial)
            elapsed  = now - self.start_times[vec_id]
            num_edges = len(self.neighborhoods[vec_id])

            bar   = '█' * int(progress * 20) + '░' * (20 - int(progress * 20))
            state = self.point_state.get(vec_id, 'UPDATE')

            lines.append(
                f"  {vec_id:>12d}  [{bar}] {progress*100:6.2f}%  "
                f"uncov={uncov:>14,}  edges={num_edges:>5} "
                f"elapsed={elapsed:>5.0f}s  {state}"
            )

        print('\n'.join(lines), flush=True)

    def sendMessages(self, compute_distances, message):
        if compute_distances:
            vecs   = self._fetch_vecs(compute_distances)           # (k, d)
            norms_ = np.einsum('ij,ij->i', vecs, vecs)            # (k,) squared norms on-the-fly
        else:
            d      = self.h5_dataset.shape[1]
            vecs   = np.empty((0, d),  dtype=np.float32)
            norms_ = np.empty((0,),    dtype=np.float32)

        t0            = time.time()
        futures       = [w.message.remote(vecs, norms_, compute_distances, message) for w in self.workers]
        all_responses = ray.get(futures)
        rtt           = time.time() - t0

        uncov_totals = collections.defaultdict(int)
        for _, uncov_counts in all_responses:
            for vid, count in uncov_counts.items():
                uncov_totals[vid] += count

        responses = collections.defaultdict(list)
        for worker_responses, _ in all_responses:
            for resp in worker_responses:
                if resp[1] is not None:
                    heappush(responses[resp[0]], (resp[2], resp[1]))

        return {vid: responses[vid][0] for vid in responses}, rtt, dict(uncov_totals)

    def writeNeighborhood(self, vec_id):
        with open(self._computed_path(), 'a+') as f:
            f.write(f"{vec_id}\n")
        with open(self._adj_path(), 'a+') as f:
            f.write(f"{self.neighborhoods[vec_id]}\n")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, shard_id, hdf5_path, hdf5_key, num_shards, cpus):
        import socket
        from threadpoolctl import threadpool_limits, threadpool_info

        os.environ["OMP_NUM_THREADS"]      = str(cpus)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
        threadpool_limits(limits=cpus, user_api='blas')
        info = threadpool_info()
        print(f"[Worker {shard_id}] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
              f"os.cpu_count()={os.cpu_count()}  threadpool={info}", flush=True)

        self.id = shard_id
        print(f"[Worker {shard_id}] init start  node={socket.gethostname()}", flush=True)

        t0 = time.time()
        with h5py.File(hdf5_path, 'r') as f:
            total, d = f[hdf5_key].shape
        print(f"[Worker {shard_id}] shape check: {time.time()-t0:.2f}s  total={total:,}  d={d}", flush=True)

        shard_size = total // num_shards
        self.start = shard_id * shard_size
        self.end   = total if shard_id == num_shards - 1 else (shard_id + 1) * shard_size
        n          = self.end - self.start
        print(f"[Worker {shard_id}] shard [{self.start:,}, {self.end:,})  n={n:,}", flush=True)

        # Build augmented matrix X of shape (n, d+2):
        #   X[i] = [v_i | 1 | ||v_i||^2]
        # so that V @ X.T yields squared Euclidean distances via:
        #   ||v - x||^2 = [-2v | ||v||^2 | 1] @ [x | 1 | ||x||^2]^T
        t1     = time.time()
        self.X = np.empty((n, d + 2), dtype=np.float32)
        with h5py.File(hdf5_path, 'r') as f:
            self.X[:, :d] = f[hdf5_key][self.start:self.end].astype(np.float32)
        print(f"[Worker {shard_id}] vectors loaded: {time.time()-t1:.2f}s", flush=True)

        t2 = time.time()
        sq_norms         = np.einsum('ij,ij->i', self.X[:, :d], self.X[:, :d])
        self.X[:, d]     = 1.0
        self.X[:, d + 1] = sq_norms
        print(f"[Worker {shard_id}] norms computed: {time.time()-t2:.2f}s", flush=True)
        print(f"[Worker {shard_id}] init done: total={time.time()-t0:.2f}s", flush=True)

        self.n             = n
        self.active_ids    = {}
        self.free_rows     = []
        self.dists_matrix  = None
        self.uncov_indices = {}   # row -> int32 array of uncovered local indices

    def ready(self):
        return True

    def _alloc_row(self, vec_id):
        row = self.free_rows.pop() if self.free_rows else len(self.active_ids)
        self.active_ids[vec_id] = row
        return row

    def message(self, vecs, norms, vec_ids, inputs):
        t_msg = time.time()
        cmds  = [cmd for _, cmd, _ in inputs]
        print(f"[Worker {self.id}] message() start  vecs={len(vec_ids)}  "
              f"cmds={dict(zip(*np.unique(cmds, return_counts=True)))}", flush=True)

        # Pass 1: INIT and KILL
        for vec_id, command, _ in inputs:
            if command == 'INIT':
                row = self._alloc_row(vec_id)
                if self.dists_matrix is None or row >= self.dists_matrix.shape[0]:
                    capacity = max(64, row + 1)
                    new_d = np.full((capacity, self.n), np.inf, dtype=np.float32)
                    if self.dists_matrix is not None:
                        new_d[:self.dists_matrix.shape[0]] = self.dists_matrix
                    self.dists_matrix = new_d
                ui = np.arange(self.n, dtype=np.int32)
                if self.start <= vec_id < self.end:
                    ui = ui[ui != (vec_id - self.start)]
                self.uncov_indices[row] = ui
            elif command == 'KILL':
                row = self.active_ids.pop(vec_id)
                self.free_rows.append(row)
                del self.uncov_indices[row]

        # Compute distances — full matmul if any INIT, sparse matmul if pure UPDATE
        D, union, vec_id_to_col, is_sparse = self._compute_dist(vecs, norms, vec_ids, inputs)

        # Pass 2: update uncovered sets, then pick next neighbor
        for vec_id, command, update_vec_id in inputs:
            if command != 'UPDATE' or D is None:
                continue
            row = self.active_ids[vec_id]
            col = vec_id_to_col[update_vec_id]
            ui  = self.uncov_indices[row]
            if len(ui) == 0:
                continue
            if is_sparse:
                d_col_at_ui = D[col][np.searchsorted(union, ui)]
            else:
                d_col_at_ui = D[col][ui]
            keep = self.dists_matrix[row][ui] <= d_col_at_ui
            self.uncov_indices[row] = ui[keep]

        response = []
        for vec_id, command, _ in inputs:
            if command == 'KILL':
                continue
            row = self.active_ids[vec_id]
            ui  = self.uncov_indices[row]
            if len(ui) > 0:
                row_dists = self.dists_matrix[row] if len(ui) == self.n else self.dists_matrix[row][ui]
                local_idx = int(np.argmin(row_dists))
                rv        = int(ui[local_idx])
                dist      = float(self.dists_matrix[row][rv])
                response.append((vec_id, rv + self.start, dist))
            else:
                response.append((vec_id, None, None))

        uncov_counts = {
            vec_id: len(self.uncov_indices[self.active_ids[vec_id]])
            for vec_id, command, _ in inputs
            if command != 'KILL'
        }

        print(f"[Worker {self.id}] message() done  elapsed={time.time()-t_msg:.2f}s", flush=True)
        return response, uncov_counts

    SPARSE_THRESHOLD = 5_000_000

    def _compute_dist(self, vecs, norms, vec_ids, inputs):
        if not vec_ids:
            return None, None, {}, False

        V = np.hstack([
            -2 * vecs,
            norms[:, None],
            np.ones((len(vec_ids), 1), dtype=np.float32),
        ])

        vec_id_to_col = {vec_id: i for i, vec_id in enumerate(vec_ids)}
        init_ids      = [v for v, cmd, _ in inputs if cmd == 'INIT']

        if init_ids:
            print(f"[Worker {self.id}] _compute_dist INIT  shape=({len(vec_ids)}, {self.n})", flush=True)
            t0 = time.time()
            D  = V @ self.X.T  # (k, shard_size)
            print(f"[Worker {self.id}] _compute_dist INIT matmul done  elapsed={time.time()-t0:.2f}s", flush=True)
            for vec_id in init_ids:
                row = self.active_ids.get(vec_id)
                if row is not None:
                    self.dists_matrix[row] = D[vec_id_to_col[vec_id]]
            return D, None, vec_id_to_col, False

        # Pure UPDATE round: compute union of uncovered indices across all active rows
        update_rows = [self.active_ids[v] for v, cmd, _ in inputs if cmd == 'UPDATE']
        if not update_rows:
            return None, None, vec_id_to_col, False

        union_mask = np.zeros(self.n, dtype=np.bool_)
        for row in update_rows:
            ui = self.uncov_indices[row]
            if len(ui) > 0:
                union_mask[ui] = True
        union = np.where(union_mask)[0].astype(np.int32)

        if len(union) < self.SPARSE_THRESHOLD:
            print(f"[Worker {self.id}] _compute_dist SPARSE  union={len(union):,}  "
                  f"shape=({len(vec_ids)}, {len(union)})", flush=True)
            t0    = time.time()
            D_sub = V @ self.X[union].T
            print(f"[Worker {self.id}] _compute_dist SPARSE matmul done  elapsed={time.time()-t0:.2f}s", flush=True)
            return D_sub, union, vec_id_to_col, True

        print(f"[Worker {self.id}] _compute_dist DENSE  union={len(union):,}  "
              f"shape=({len(vec_ids)}, {self.n})", flush=True)
        t0 = time.time()
        D  = V @ self.X.T
        print(f"[Worker {self.id}] _compute_dist DENSE matmul done  elapsed={time.time()-t0:.2f}s", flush=True)
        return D, None, vec_id_to_col, False


# ---------------------------------------------------------------------------
# Ray actors
# ---------------------------------------------------------------------------

@ray.remote
class WorkerActor(Worker):
    pass


@ray.remote
class CoordinatorActor(Coordinator):
    def __init__(self, hdf5_path, hdf5_key, name, save_path, num_points, batch, workers):
        self.workers = workers
        super().__init__(hdf5_path, hdf5_key, name, save_path, num_points, batch)


if __name__ == '__main__':
    main()
