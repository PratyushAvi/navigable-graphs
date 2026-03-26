# test.py

import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["OPENBLAS_NUM_THREADS"] = "16"

import numpy as np
import random
import time
import ray

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, worker_id, EFS_PATH, num_shards):
        self.id       = worker_id
        self.EFS_PATH = EFS_PATH
        print(f"[Worker {worker_id}] init start", flush=True)

        t0    = time.time()
        total = np.load(f"{self.EFS_PATH}/vectors.npy", mmap_mode='r').shape[0]
        print(f"[Worker {worker_id}] shape check: {time.time()-t0:.2f}s  total={total:,}", flush=True)

        shard_size   = total // num_shards
        self.start   = worker_id * shard_size
        self.end     = total if worker_id == num_shards - 1 else (worker_id + 1) * shard_size
        print(f"[Worker {worker_id}] shard [{self.start:,}, {self.end:,})  n={self.end-self.start:,}", flush=True)

        # Load shard slice into X then close mmaps
        t1      = time.time()
        dataset = np.load(f"{self.EFS_PATH}/vectors.npy",  mmap_mode='r')
        norms   = np.load(f"{self.EFS_PATH}/sq_norms.npy", mmap_mode='r')
        print(f"[Worker {worker_id}] mmap open: {time.time()-t1:.2f}s", flush=True)

        n        = self.end - self.start
        self.X   = np.empty((n, 102), dtype=np.float32)
        t2       = time.time()
        self.X[:, :100] = dataset[self.start:self.end]
        print(f"[Worker {worker_id}] vectors loaded: {time.time()-t2:.2f}s", flush=True)
        t3       = time.time()
        self.X[:, 100]  = 1.0
        self.X[:, 101]  = norms[self.start:self.end]
        print(f"[Worker {worker_id}] norms loaded: {time.time()-t3:.2f}s", flush=True)
        del dataset, norms
        print(f"[Worker {worker_id}] init done: total={time.time()-t0:.2f}s", flush=True)

        self.n             = n
        self.active_ids    = {}
        self.free_rows     = []
        self.dists_matrix  = None
        self.uncov_indices = {}   # row -> int32 array of uncovered local indices

    def _alloc_row(self, vec_id):
        row = self.free_rows.pop() if self.free_rows else len(self.active_ids)
        self.active_ids[vec_id] = row
        return row

    def message(self, vecs, norms, vec_ids, inputs):
        print(inputs, " received", flush=True)
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

        # Compute distances — always compute, store only for INIT
        t0 = time.time()
        D, vec_id_to_col = self._compute_dist(vecs, norms, vec_ids, inputs)
        compute_time = time.time() - t0

        # --- mask update: filter uncov_indices per row ---
        t_mask0 = time.time()
        for vec_id, command, update_vec_id in inputs:
            if command != 'UPDATE' or D is None:
                continue
            row = self.active_ids[vec_id]
            col = vec_id_to_col[update_vec_id]
            ui  = self.uncov_indices[row]
            if len(ui) == 0:
                continue
            # keep indices where query is closer than (or equal to) the neighbor
            keep = self.dists_matrix[row][ui] <= D[col][ui]
            self.uncov_indices[row] = ui[keep]
        t_mask_total = time.time() - t_mask0

        # --- argmin: find nearest uncovered element per row ---
        t_argmin0 = time.time()
        response = []
        for vec_id, command, _ in inputs:
            if command == 'KILL':
                continue
            row = self.active_ids[vec_id]
            ui  = self.uncov_indices[row]
            if len(ui) > 0:
                local_idx = int(np.argmin(self.dists_matrix[row][ui]))
                rv        = int(ui[local_idx])
                dist      = float(self.dists_matrix[row][rv])
                response.append((vec_id, rv + self.start, dist))
            else:
                response.append((vec_id, None, None))
        t_argmin_total = time.time() - t_argmin0

        n_updates  = sum(1 for _, cmd, _ in inputs if cmd == 'UPDATE')
        n_queries  = sum(1 for _, cmd, _ in inputs if cmd != 'KILL')
        avg_uncov  = np.mean([len(self.uncov_indices[self.active_ids[v]])
                              for v, cmd, _ in inputs if cmd != 'KILL']) if n_queries else 0
        print(
            f"[Worker {self.id}] pass2 ({n_updates} updates, {n_queries} argmins, "
            f"avg_uncov={avg_uncov:.0f}): "
            f"mask={t_mask_total*1000:.1f}ms  "
            f"argmin={t_argmin_total*1000:.1f}ms  "
            f"total={(t_mask_total + t_argmin_total)*1000:.1f}ms",
            flush=True
        )

        return response, compute_time

    def _compute_dist(self, vecs, norms, vec_ids, inputs):
        if not vec_ids:
            return None, {}

        t0 = time.time()
        V = np.hstack([
            -2 * vecs,
            norms[:, None],
            np.ones((len(vec_ids), 1), dtype=np.float32),
        ])
        t_build_v = time.time() - t0

        t1 = time.time()
        D = V @ self.X.T  # (k, shard_size)
        t_matmul = time.time() - t1

        # Map vec_id -> column index in D
        vec_id_to_col = {vec_id: i for i, vec_id in enumerate(vec_ids)}

        # Store into dists_matrix only for INIT vecs
        t2 = time.time()
        init_ids = [v for v, cmd, _ in inputs if cmd == 'INIT']
        if init_ids:
            for vec_id in init_ids:
                row = self.active_ids.get(vec_id)
                if row is not None:
                    self.dists_matrix[row] = D[vec_id_to_col[vec_id]]
        t_store = time.time() - t2

        print(
            f"[Worker {self.id}] _compute_dist ({len(vec_ids)} vecs): "
            f"build_V={t_build_v*1000:.1f}ms  "
            f"matmul={t_matmul*1000:.1f}ms  "
            f"store={t_store*1000:.1f}ms",
            flush=True
        )

        return D, vec_id_to_col


@ray.remote
class WorkerActor(Worker):
    pass


def main():
    import argparse
    os.environ["OMP_NUM_THREADS"] = "16"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True,       help="Path to dataset directory")
    parser.add_argument("--num_shards",  type=int, default=17)
    parser.add_argument("--num_workers", type=int, default=17)
    parser.add_argument("--cpus",        type=int, default=16)
    args = parser.parse_args()

    assert args.num_workers <= args.num_shards, \
        f"num_workers ({args.num_workers}) must be <= num_shards ({args.num_shards})"

    ray.init(address="auto", runtime_env={"env_vars": {"OMP_NUM_THREADS": str(args.cpus), "OPENBLAS_NUM_THREADS": str(args.cpus)}})

    expected_cpus = args.num_workers * args.cpus
    print(f"Waiting for {args.num_workers} workers ({expected_cpus} CPUs)...", flush=True)
    while True:
        available = ray.cluster_resources().get("CPU", 0)
        print(f"  CPUs available: {available:.0f}/{expected_cpus}", flush=True)
        if available >= expected_cpus:
            break
        time.sleep(15)
    print("All workers ready.", flush=True)

    workers = [WorkerActor.options(num_cpus=args.cpus).remote(i, args.data, args.num_shards)
               for i in range(args.num_workers)]
    dataset = np.load(f"{args.data}/vectors.npy",  mmap_mode='r')
    norms   = np.load(f"{args.data}/sq_norms.npy", mmap_mode='r')

    batch_sizes = [50]
    num_trials  = 3

    print(f"{'Operation':<12} {'Batch Size':<12} {'Trial':<8} {'Wall (s)':<14} {'Compute (s)':<14}", flush=True)
    print("-" * 62, flush=True)

    for batch_size in batch_sizes:
        for trial in range(num_trials):
            vec_ids = random.sample(range(dataset.shape[0]), batch_size)
            vecs    = dataset[vec_ids].astype(np.float32)
            norms_  = norms[vec_ids].astype(np.float32)
            inputs  = [(i, 'INIT', None) for i in vec_ids]

            wall_start   = time.time()
            futures      = [w.message.remote(vecs, norms_, vec_ids, inputs) for w in workers]
            all_results  = ray.get(futures)
            wall_time    = time.time() - wall_start
            compute_times = [r[1] for r in all_results]
            print(f"{'INIT':<12} {batch_size:<12} {trial:<8} {wall_time:<14.4f} avg={np.mean(compute_times):.4f} max={np.max(compute_times):.4f}", flush=True)

            compute_vecs = [i + 1 for i in vec_ids]
            update_vecs  = dataset[compute_vecs].astype(np.float32)
            update_norms = norms[compute_vecs].astype(np.float32)
            inputs       = [(i, 'UPDATE', i + 1) for i in vec_ids]

            wall_start   = time.time()
            futures      = [w.message.remote(update_vecs, update_norms, compute_vecs, inputs) for w in workers]
            all_results  = ray.get(futures)
            wall_time    = time.time() - wall_start
            compute_times = [r[1] for r in all_results]
            print(f"{'UPDATE':<12} {batch_size:<12} {trial:<8} {wall_time:<14.4f} avg={np.mean(compute_times):.4f} max={np.max(compute_times):.4f}", flush=True)

            inputs = [(i, 'KILL', None) for i in vec_ids]

            wall_start  = time.time()
            futures     = [w.message.remote(
                np.empty((0, 100), dtype=np.float32),
                np.empty((0,),     dtype=np.float32),
                [], inputs
            ) for w in workers]
            ray.get(futures)
            wall_time = time.time() - wall_start
            print(f"{'KILL':<12} {batch_size:<12} {trial:<8} {wall_time:<14.4f}", flush=True)

        print("", flush=True)

if __name__ == "__main__":
    main()
