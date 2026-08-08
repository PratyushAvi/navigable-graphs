import numpy as np
import collections
import time
from heapq import heappush
import ray
import os

os.environ["RAY_DEDUP_LOGS"] = "0"

def main():
    import argparse

    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Path to dataset directory holding vectors.npy, ids.npy, sq_norms.npy")
    source.add_argument("--hdf5", help="Path to an HDF5 dataset file (alternative to --data)")
    parser.add_argument("--hdf5_key", default="train", help="Key inside the HDF5 file (default: train)")
    parser.add_argument("--save_path", required=True, help="Path to save results")
    parser.add_argument("--dataset", required=True, help="Dataset name, used for output/checkpoint filenames")
    parser.add_argument("--num_points", type=int, default=10_000,
                        help="Number of neighborhoods to compute (0 = all points in dataset)")
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--num_shards", type=int, default=17)
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--coordinator_cpus", type=int, default=1)
    parser.add_argument("--metric", default="euclidean",
                        help="Metric label for output filenames, kept as the final '-' segment "
                             "to match simulrun.py (default: euclidean). The prune itself is "
                             "squared-Euclidean; this does not change the computation.")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="alpha-reachability parameter (>= 1). A point p counts as covered "
                             "by edge (u, v) when d(v, p) < d(u, p) / alpha, so larger alpha "
                             "requires more progress per edge and produces denser neighborhoods. "
                             "alpha=1 (the default) is the standard coverage rule.")
    args = parser.parse_args()

    if args.alpha < 1.0:
        parser.error(f"--alpha must be >= 1, got {args.alpha}")

    source_spec = VectorSource(args.data, args.hdf5, args.hdf5_key)

    # num_points == 0 means "the whole dataset"
    if args.num_points == 0:
        args.num_points = source_spec.total()
        print(f"num_points set to full dataset size: {args.num_points:,}", flush=True)

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
    workers = [WorkerActor.options(num_cpus=args.cpus).remote(i, source_spec, args.num_shards, args.cpus, args.alpha)
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
        source=source_spec,
        SAVE_PATH=args.save_path,
        dataset=args.dataset,
        num_points=args.num_points,
        batch=args.batch,
        workers=workers,
        alpha=args.alpha,
        metric=args.metric,
    )
    print("Waiting for coordinator to initialize...", flush=True)
    ray.get(coordinator.ready.remote())
    print("Coordinator ready. Starting computation.", flush=True)

    ray.get(coordinator.computeNeighborhoods.remote())


# ---------------------------------------------------------------------------
# Vector source
# ---------------------------------------------------------------------------

class VectorSource:
    """Describes where vectors come from, abstracting .npy-directory vs HDF5.

    Holds only paths, so it pickles cleanly and each Ray actor opens its own
    handles. The two backends differ in three ways:
      - .npy has explicit ids.npy; HDF5 ids are implicit 0..total-1
      - .npy has precomputed sq_norms.npy; HDF5 norms are computed on load
      - h5py fancy indexing needs sorted, duplicate-free indices (see fetch)
    """

    def __init__(self, data_dir=None, hdf5_path=None, hdf5_key="train"):
        self.data_dir  = data_dir
        self.hdf5_path = hdf5_path
        self.hdf5_key  = hdf5_key
        self.is_hdf5   = hdf5_path is not None

    def shape(self):
        if self.is_hdf5:
            import h5py
            with h5py.File(self.hdf5_path, 'r') as f:
                total, dim = f[self.hdf5_key].shape
            return int(total), int(dim)
        vshape = np.load(f"{self.data_dir}/vectors.npy", mmap_mode='r').shape
        return int(vshape[0]), int(vshape[1])

    def total(self):
        return self.shape()[0]

    def open_reader(self):
        """Return a _Reader for random access to individual vectors (coordinator)."""
        return _Hdf5Reader(self) if self.is_hdf5 else _NpyReader(self)

    def load_shard(self, start, end, dim):
        """Return (vectors, sq_norms) for rows [start, end) as float32."""
        if self.is_hdf5:
            import h5py
            with h5py.File(self.hdf5_path, 'r') as f:
                vecs = f[self.hdf5_key][start:end].astype(np.float32)
            # No precomputed norms in HDF5 — derive them from the shard itself.
            return vecs, np.einsum('ij,ij->i', vecs, vecs)
        dataset = np.load(f"{self.data_dir}/vectors.npy",  mmap_mode='r')
        norms   = np.load(f"{self.data_dir}/sq_norms.npy", mmap_mode='r')
        return dataset[start:end], norms[start:end]


class _NpyReader:
    def __init__(self, source):
        self.vector_ids = np.load(f"{source.data_dir}/ids.npy",      mmap_mode='r')
        self.vectors    = np.load(f"{source.data_dir}/vectors.npy",  mmap_mode='r')
        self.norms      = np.load(f"{source.data_dir}/sq_norms.npy", mmap_mode='r')
        self.dim        = self.vectors.shape[1]

    def fetch(self, indices):
        vecs   = self.vectors[indices].astype(np.float32)
        norms_ = self.norms[indices].astype(np.float32)
        return vecs, norms_


class _Hdf5Reader:
    def __init__(self, source):
        import h5py
        self._file      = h5py.File(source.hdf5_path, 'r')
        self.h5_dataset = self._file[source.hdf5_key]
        total, dim      = self.h5_dataset.shape
        self.dim        = int(dim)
        # IDs are implicit for HDF5 inputs.
        self.vector_ids = np.arange(total, dtype=np.int64)

    def fetch(self, indices):
        # h5py fancy indexing requires strictly increasing indices with no
        # duplicates. np.unique returns sorted uniques; inverse restores the
        # caller's original order (and re-expands any duplicates).
        idx = np.asarray(indices, dtype=np.int64)
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        vecs = self.h5_dataset[unique_idx.tolist()].astype(np.float32)[inverse]
        return vecs, np.einsum('ij,ij->i', vecs, vecs).astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    def __init__(self, source, SAVE_PATH, dataset, num_points, batch, alpha=1.0,
                 metric="euclidean"):
        os.environ["RAY_DEDUP_LOGS"] = "0"
        self.source     = source
        self.SAVE_PATH  = SAVE_PATH
        self.dataset    = dataset
        self.num_points = num_points
        self.batch      = batch
        self.alpha      = alpha
        self.metric     = metric

        # Filenames match simulrun.py: "<dataset>-alpha<a>-<metric>". Each alpha
        # builds a different graph so it always appears, including alpha=1, and
        # '.' would break the '.txt' handling in the analysis scripts'
        # parse_filename, so 1.2 -> "alpha1p2". The metric stays the final '-'
        # segment, which is what parse_filename relies on.
        tag = "-alpha" + f"{alpha:g}".replace('.', 'p') + f"-{metric}"
        self.computed_path = f"{self.SAVE_PATH}/{self.dataset}{tag}-computed.txt"
        self.adj_list_path = f"{self.SAVE_PATH}/adj-list-{self.dataset}{tag}.txt"

        self.reader     = source.open_reader()
        self.vector_ids = self.reader.vector_ids
        self.dim        = self.reader.dim

        self.neighborhoods = {}
        self.active        = set()
        self.computed      = set()
        self.start_times   = {}
        self.round_num     = 0
        self.rtt_history   = []
        self.uncov_initial = {}   # vec_id -> total uncov after INIT
        self.uncov_current = {}   # vec_id -> current total uncov
        self.point_state   = {}   # vec_id -> 'INIT' | 'UPDATE'

        if os.path.exists(self.computed_path):
            with open(self.computed_path, "r") as f:
                for p in f.readlines():
                    self.computed.add(int(p.strip()))

        print(f"alpha-reachability: alpha={alpha:g}  ->  {self.adj_list_path}", flush=True)
        print(f"Resuming from {len(self.computed)} already computed neighborhoods.", flush=True)

        need       = self.num_points - len(self.computed)
        n_sample   = min(need * 10, len(self.vector_ids))
        candidates = np.random.choice(self.vector_ids, n_sample, replace=False)
        filtered   = [int(v) for v in candidates if v not in self.computed]
        self.queue = collections.deque(filtered[:need])
        print(f"Queue built with {len(self.queue)} uncomputed vectors.", flush=True)

    def ready(self):
        return True

    def computeNeighborhoods(self):
        print("[Coordinator] computeNeighborhoods start", flush=True)
        self.active = set(self.queue.popleft() for _ in range(self.batch))
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
            # neighbor chosen for each vec_id this round; its (neighbor, uncov)
            # tuple is recorded after sendMessages refreshes uncov_current, so
            # the stored uncov reflects the edge *after* it takes effect.
            edge_this_round = {}

            for vec_id in list(self.active):
                if vec_id not in responses or responses[vec_id] is None:
                    message.append((vec_id, 'KILL', None))
                    self.writeNeighborhood(vec_id)
                    self.active.remove(vec_id)
                    self.computed.add(vec_id)
                    self.uncov_initial.pop(vec_id, None)
                    self.uncov_current.pop(vec_id, None)
                    self.point_state.pop(vec_id, None)

                    if self.queue:
                        new_vec = self.queue.popleft()
                        self.active.add(new_vec)
                        message.append((new_vec, 'INIT', None))
                        compute_distances.append(new_vec)
                        self.neighborhoods[new_vec] = []
                        self.start_times[new_vec]   = time.time()
                        self.point_state[new_vec]   = 'INIT'
                else:
                    _, neighbor = responses[vec_id]
                    edge_this_round[vec_id] = neighbor
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

            # Now uncov_current reflects this round's edges. Record each edge with
            # the uncov *after* it took effect (the neighborhood is a tuple of
            # (neighbor, points uncovered)).
            for vec_id, neighbor in edge_this_round.items():
                self.neighborhoods[vec_id].append((neighbor, self.uncov_current[vec_id]))

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
            vecs, norms_ = self.reader.fetch(compute_distances)
        else:
            vecs   = np.empty((0, self.dim), dtype=np.float32)
            norms_ = np.empty((0,),          dtype=np.float32)

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
        with open(self.computed_path, 'a+') as f:
            f.write(f"{vec_id}\n")
        with open(self.adj_list_path, 'a+') as f:
            f.write(f"{self.neighborhoods[vec_id]}\n")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, shard_id, source, num_shards, cpus, alpha=1.0):
        import os
        from threadpoolctl import threadpool_limits, threadpool_info
        os.environ["OMP_NUM_THREADS"]      = str(cpus)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
        threadpool_limits(limits=cpus, user_api='blas')
        info = threadpool_info()
        print(f"[Worker {shard_id}] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}  "
              f"os.cpu_count()={os.cpu_count()}  threadpool={info}", flush=True)

        import socket
        self.id     = shard_id
        self.source = source
        # alpha-reachability: waypoint w covers p only when d(w, p) < d(source, p) / alpha.
        # Distances here are squared euclidean, so the test scales by alpha^2.
        # alpha = 1 reproduces the standard coverage rule.
        if alpha < 1.0:
            raise ValueError(f"alpha must be >= 1, got {alpha}")
        self.alpha_sq = float(alpha) ** 2
        print(f"[Worker {shard_id}] init start  node={socket.gethostname()}", flush=True)

        t0         = time.time()
        total, dim = source.shape()
        self.dim   = dim
        print(f"[Worker {shard_id}] shape check: {time.time()-t0:.2f}s  total={total:,}  dim={dim}", flush=True)

        shard_size   = total // num_shards
        self.start   = shard_id * shard_size
        self.end     = total if shard_id == num_shards - 1 else (shard_id + 1) * shard_size
        print(f"[Worker {shard_id}] shard [{self.start:,}, {self.end:,})  n={self.end-self.start:,}", flush=True)

        # Build augmented matrix X of shape (n, dim+2):
        #   X[i] = [v_i | 1 | ||v_i||^2]
        # so that V @ X.T yields squared Euclidean distances via:
        #   ||v - x||^2 = [-2v | ||v||^2 | 1] @ [x | 1 | ||x||^2]^T
        t1           = time.time()
        vecs, norms  = source.load_shard(self.start, self.end, dim)
        print(f"[Worker {shard_id}] shard read: {time.time()-t1:.2f}s", flush=True)

        n      = self.end - self.start
        self.X = np.empty((n, dim + 2), dtype=np.float32)
        t2     = time.time()
        self.X[:, :dim] = vecs
        print(f"[Worker {shard_id}] vectors loaded: {time.time()-t2:.2f}s", flush=True)
        t3     = time.time()
        self.X[:, dim]     = 1.0
        self.X[:, dim + 1] = norms
        print(f"[Worker {shard_id}] norms loaded: {time.time()-t3:.2f}s", flush=True)
        del vecs, norms
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
            # Keep p uncovered unless the waypoint covers it. With alpha the covering
            # test is d(w, p) * alpha < d(source, p), so p stays uncovered when
            # d(source, p) <= d(w, p) * alpha_sq (squared distances throughout).
            if is_sparse:
                # ui is a sorted subset of union; searchsorted maps ui -> positions in D
                d_col_at_ui = D[col][np.searchsorted(union, ui)]
                keep = self.dists_matrix[row][ui] <= d_col_at_ui * self.alpha_sq
                ui   = ui[keep]
            elif len(ui) == self.n:
                # ui == arange(n): avoid expensive fancy indexing, compare directly
                keep = self.dists_matrix[row] <= D[col] * self.alpha_sq
                ui   = np.where(keep)[0].astype(np.int32)
            else:
                d_col_at_ui = D[col][ui]
                keep = self.dists_matrix[row][ui] <= d_col_at_ui * self.alpha_sq
                ui   = ui[keep]
            # Explicitly remove the waypoint itself — the pruning condition
            # dist(source, w) <= dist(w, w)=0 is True when they are duplicates
            # (dist=0), so the waypoint would otherwise never leave uncov.
            local_wp = update_vec_id - self.start
            if 0 <= local_wp < self.n:
                ui = ui[ui != local_wp]
            self.uncov_indices[row] = ui

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
    def __init__(self, source, SAVE_PATH, dataset, num_points, batch, workers, alpha=1.0,
                 metric="euclidean"):
        self.workers = workers
        super().__init__(source, SAVE_PATH, dataset, num_points, batch, alpha, metric)


if __name__ == '__main__':
    main()
