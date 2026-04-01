import numpy as np
import collections
import time
from heapq import heappush
import ray
import os

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    parser.add_argument("--num_points", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--num_shards", type=int, default=17)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--coordinator_cpus", type=int, default=1)
    args = parser.parse_args()

    ray.init(address="auto", runtime_env={"env_vars": {"OMP_NUM_THREADS": str(args.cpus), "OPENBLAS_NUM_THREADS": str(args.cpus)}})

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
    workers = [WorkerActor.options(num_cpus=args.cpus).remote(i, args.data, args.num_shards, args.cpus)
               for i in range(args.num_shards)]

    pending = {w.ready.remote(): i for i, w in enumerate(workers)}
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
        EFS_PATH=args.data,
        num_points=args.num_points,
        batch=args.batch,
        workers=workers
    )
    print("Waiting for coordinator to initialize...", flush=True)
    ray.get(coordinator.ready.remote())
    print("Coordinator ready. Starting computation.", flush=True)

    ray.get(coordinator.computeNeighborhoods.remote())


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    def __init__(self, EFS_PATH, num_points, batch):
        os.environ["RAY_DEDUP_LOGS"] = "0"
        self.EFS_PATH   = EFS_PATH
        self.num_points = num_points
        self.batch      = batch

        self.vector_ids = np.load(f"{self.EFS_PATH}/ids.npy",      mmap_mode='r')
        self.dataset    = np.load(f"{self.EFS_PATH}/vectors.npy",   mmap_mode='r')
        self.norms      = np.load(f"{self.EFS_PATH}/sq_norms.npy",  mmap_mode='r')

        self.neighborhoods = {}
        self.active        = set()
        self.computed      = set()
        self.start_times   = {}
        self.round_num     = 0
        self.rtt_history   = []
        self.uncov_initial = {}   # vec_id -> total uncov after INIT
        self.uncov_current = {}   # vec_id -> current total uncov
        self.point_state   = {}   # vec_id -> 'INIT' | 'UPDATE'

        with open(f"{self.EFS_PATH}/computed.txt", "r") as f:
            for p in f.readlines():
                self.computed.add(int(p.strip()))

        print(f"Resuming from {len(self.computed)} already computed neighborhoods.", flush=True)

    def ready(self):
        return True

    def computeNeighborhoods(self):
        print("[Coordinator] computeNeighborhoods start", flush=True)
        self.active = set(int(v) for v in np.random.choice(self.vector_ids, self.batch, replace=False))
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

                    # the neighborhood is a tuple of (neighbor, points uncovered)
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

        lines = [
            f"\n[Round {self.round_num}] Took {self.rtt_history[-1]:.1f}s |\n"
            f"RTT min={rtt_min:.1f}s avg={rtt_avg:.1f}s max={rtt_max:.1f}s  "
            f"active={len(self.active)}  completed={len(self.computed)}\n"
            f"goal ETA={(self.num_points - len(self.computed)) * rtt_avg * 1000 / self.batch:.2f}s (assuming max deg < 1000)"
        ]

        now = time.time()
        for vec_id in sorted(self.active):
            uncov     = self.uncov_current.get(vec_id, 0)
            initial   = self.uncov_initial.get(vec_id, uncov) or 1
            covered   = initial - uncov
            progress  = max(0.0, covered / initial)
            elapsed   = now - self.start_times[vec_id]
            num_edges = len(self.neighborhoods[vec_id])

            bar   = '█' * int(progress * 20) + '░' * (20 - int(progress * 20))
            # eta   = f"{elapsed * uncov / covered:.0f}s" if covered > 0 else "--"
            state = self.point_state.get(vec_id, 'UPDATE')

            lines.append(
                f"  {vec_id:>12d}  [{bar}] {progress*100:6.2f}%  "
                f"uncov={uncov:>14,}  edges={num_edges:>5} "
                f"elapsed={elapsed:>5.0f}s  {state}"
            )

        print('\n'.join(lines), flush=True)

    def sendMessages(self, compute_distances, message):
        if compute_distances:
            vecs   = self.dataset[compute_distances].astype(np.float32)
            norms_ = self.norms[compute_distances].astype(np.float32)
        else:
            vecs   = np.empty((0, 100), dtype=np.float32)
            norms_ = np.empty((0,),     dtype=np.float32)

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
        with open(f"{self.EFS_PATH}/computed.txt", 'a+') as f:
            f.write(f"{vec_id}\n")
        with open(f"{self.EFS_PATH}/neighborhoods.txt", 'a+') as f:
            f.write(f"{self.neighborhoods[vec_id]}\n")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, shard_id, EFS_PATH, num_shards, cpus):
        import os
        from threadpoolctl import threadpool_limits, threadpool_info
        os.environ["OMP_NUM_THREADS"]      = str(cpus)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
        threadpool_limits(limits=cpus, user_api='blas')
        info = threadpool_info()
        print(f"[Worker {shard_id}] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
              f"os.cpu_count()={os.cpu_count()}  threadpool={info}", flush=True)

        import socket
        self.id       = shard_id
        self.EFS_PATH = EFS_PATH
        print(f"[Worker {shard_id}] init start  node={socket.gethostname()}", flush=True)

        t0    = time.time()
        total = np.load(f"{self.EFS_PATH}/vectors.npy", mmap_mode='r').shape[0]
        print(f"[Worker {shard_id}] shape check: {time.time()-t0:.2f}s  total={total:,}", flush=True)

        shard_size   = total // num_shards
        self.start   = shard_id * shard_size
        self.end     = total if shard_id == num_shards - 1 else (shard_id + 1) * shard_size
        print(f"[Worker {shard_id}] shard [{self.start:,}, {self.end:,})  n={self.end-self.start:,}", flush=True)

        # Load shard slice into X then close mmaps
        t1      = time.time()
        dataset = np.load(f"{self.EFS_PATH}/vectors.npy",  mmap_mode='r')
        norms   = np.load(f"{self.EFS_PATH}/sq_norms.npy", mmap_mode='r')
        print(f"[Worker {shard_id}] mmap open: {time.time()-t1:.2f}s", flush=True)

        n      = self.end - self.start
        self.X = np.empty((n, 102), dtype=np.float32)
        t2     = time.time()
        self.X[:, :100] = dataset[self.start:self.end]
        print(f"[Worker {shard_id}] vectors loaded: {time.time()-t2:.2f}s", flush=True)
        t3     = time.time()
        self.X[:, 100]  = 1.0
        self.X[:, 101]  = norms[self.start:self.end]
        print(f"[Worker {shard_id}] norms loaded: {time.time()-t3:.2f}s", flush=True)
        del dataset, norms
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
        cmds = [cmd for _, cmd, _ in inputs]
        print(f"[Worker {self.id}] message() start  vecs={len(vec_ids)}  cmds={dict(zip(*np.unique(cmds, return_counts=True)))}", flush=True)

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

        # Pass 2: sparse UPDATE then sparse argmin
        for vec_id, command, update_vec_id in inputs:
            if command != 'UPDATE' or D is None:
                continue
            row = self.active_ids[vec_id]
            col = vec_id_to_col[update_vec_id]
            ui  = self.uncov_indices[row]
            if len(ui) == 0:
                continue
            if is_sparse:
                # ui is a sorted subset of union; searchsorted maps ui -> positions in D
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
            D = V @ self.X.T  # (k, shard_size)
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

        # Build union via boolean mask — O(k × shard_size) time, O(shard_size) memory
        # vs np.unique(np.concatenate(parts)) which is O(k × shard_size × log) and allocates k × shard_size elements
        union_mask = np.zeros(self.n, dtype=np.bool_)
        for row in update_rows:
            ui = self.uncov_indices[row]
            if len(ui) > 0:
                union_mask[ui] = True
        union = np.where(union_mask)[0].astype(np.int32)

        if len(union) < self.SPARSE_THRESHOLD:
            print(f"[Worker {self.id}] _compute_dist SPARSE  union={len(union):,}  shape=({len(vec_ids)}, {len(union)})", flush=True)
            t0 = time.time()
            X_sub = self.X[union]
            D_sub = V @ X_sub.T
            print(f"[Worker {self.id}] _compute_dist SPARSE matmul done  elapsed={time.time()-t0:.2f}s", flush=True)
            return D_sub, union, vec_id_to_col, True

        print(f"[Worker {self.id}] _compute_dist DENSE  union={len(union):,}  shape=({len(vec_ids)}, {self.n})", flush=True)
        t0 = time.time()
        D = V @ self.X.T
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
    def __init__(self, EFS_PATH, num_points, batch, workers):
        self.workers = workers
        super().__init__(EFS_PATH, num_points, batch)


if __name__ == '__main__':
    main()
