"""
mpi4py variant of distributed_robust_prune.py

Launch with SLURM:
    #SBATCH --ntasks=<num_shards+1>   # 1 coordinator + num_shards workers
    #SBATCH --cpus-per-task=16
    srun python mpi4py_distributed_robust_prune.py --data /path/to/data --num_points 10000 --batch 50

Rank 0  = coordinator
Ranks 1..num_shards = workers
"""

import numpy as np
import collections
import time
import argparse
from heapq import heappush
from mpi4py import MPI


# ---------------------------------------------------------------------------
# MPI tags
# ---------------------------------------------------------------------------
TAG_MESSAGE  = 1   # coordinator -> worker: (vecs, norms, vec_ids, inputs)
TAG_RESPONSE = 2   # worker -> coordinator: (response, uncov_counts)
TAG_SHUTDOWN = 3   # coordinator -> worker: None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, shard_id, data_path, num_shards, cpus):
        import os
        os.environ["OMP_NUM_THREADS"]      = str(cpus)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)

        self.id       = shard_id
        print(f"[Worker {shard_id}] init start", flush=True)

        t0    = time.time()
        total = np.load(f"{data_path}/vectors.npy", mmap_mode='r').shape[0]
        print(f"[Worker {shard_id}] total={total:,}", flush=True)

        shard_size = total // num_shards
        self.start  = shard_id * shard_size
        self.end    = total if shard_id == num_shards - 1 else (shard_id + 1) * shard_size

        dataset = np.load(f"{data_path}/vectors.npy",  mmap_mode='r')
        norms   = np.load(f"{data_path}/sq_norms.npy", mmap_mode='r')

        n      = self.end - self.start
        self.X = np.empty((n, 102), dtype=np.float32)
        self.X[:, :100] = dataset[self.start:self.end]
        self.X[:, 100]  = 1.0
        self.X[:, 101]  = norms[self.start:self.end]
        del dataset, norms

        self.n             = n
        self.active_ids    = {}
        self.free_rows     = []
        self.dists_matrix  = None
        self.uncov_indices = {}   # row -> int32 array of uncovered local indices
        print(f"[Worker {shard_id}] init done: {time.time()-t0:.2f}s  shard=[{self.start:,}, {self.end:,})", flush=True)

    def _alloc_row(self, vec_id):
        row = self.free_rows.pop() if self.free_rows else len(self.active_ids)
        self.active_ids[vec_id] = row
        return row

    def message(self, vecs, norms, vec_ids, inputs):
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
                local_idx = int(np.argmin(self.dists_matrix[row][ui]))
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
            D = V @ self.X.T  # (k, shard_size)
            for vec_id in init_ids:
                row = self.active_ids.get(vec_id)
                if row is not None:
                    self.dists_matrix[row] = D[vec_id_to_col[vec_id]]
            return D, None, vec_id_to_col, False

        update_rows = [self.active_ids[v] for v, cmd, _ in inputs if cmd == 'UPDATE']
        if not update_rows:
            return None, None, vec_id_to_col, False

        # Build union via boolean mask — avoids O(k*n*log) sort from concatenate+unique
        union_mask = np.zeros(self.n, dtype=np.bool_)
        for row in update_rows:
            ui = self.uncov_indices[row]
            if len(ui) > 0:
                union_mask[ui] = True
        union = np.where(union_mask)[0].astype(np.int32)

        if len(union) < self.SPARSE_THRESHOLD:
            X_sub = self.X[union]
            D_sub = V @ X_sub.T
            return D_sub, union, vec_id_to_col, True

        D = V @ self.X.T
        return D, None, vec_id_to_col, False


# ---------------------------------------------------------------------------
# Worker event loop (runs on all ranks except 0)
# ---------------------------------------------------------------------------

def worker_loop(comm, rank, args):
    shard_id = rank - 1
    worker   = Worker(shard_id, args.data, args.num_shards, args.cpus)
    print(f"[Worker {shard_id}] ready, waiting for messages", flush=True)

    while True:
        payload = comm.recv(source=0, tag=MPI.ANY_TAG, status=(status := MPI.Status()))
        if status.Get_tag() == TAG_SHUTDOWN:
            print(f"[Worker {shard_id}] shutting down", flush=True)
            break

        vecs, norms, vec_ids, inputs = payload
        response, uncov_counts = worker.message(vecs, norms, vec_ids, inputs)
        comm.send((response, uncov_counts), dest=0, tag=TAG_RESPONSE)


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    def __init__(self, comm, data_path, num_points, batch, num_shards):
        self.comm       = comm
        self.num_shards = num_shards
        self.EFS_PATH   = data_path
        self.num_points = num_points
        self.batch      = batch

        self.vector_ids = np.load(f"{data_path}/ids.npy",      mmap_mode='r')
        self.dataset    = np.load(f"{data_path}/vectors.npy",   mmap_mode='r')
        self.norms      = np.load(f"{data_path}/sq_norms.npy",  mmap_mode='r')

        self.neighborhoods = {}
        self.active        = set()
        self.computed      = set()
        self.start_times   = {}
        self.round_num     = 0
        self.rtt_history   = []
        self.uncov_initial = {}
        self.uncov_current = {}
        self.point_state   = {}

        try:
            with open(f"{self.EFS_PATH}/computed.txt", "r") as f:
                for p in f.readlines():
                    self.computed.add(int(p.strip()))
        except FileNotFoundError:
            pass
        print(f"Resuming from {len(self.computed)} already computed neighborhoods.", flush=True)

    def computeNeighborhoods(self):
        self.active = set(int(v) for v in np.random.choice(self.vector_ids, self.batch, replace=False))
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
                        while new_vec in self.computed or new_vec in self.active:
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
        self._shutdown_workers()

    def sendMessages(self, compute_distances, message):
        if compute_distances:
            vecs   = self.dataset[compute_distances].astype(np.float32)
            norms_ = self.norms[compute_distances].astype(np.float32)
        else:
            vecs   = np.empty((0, 100), dtype=np.float32)
            norms_ = np.empty((0,),     dtype=np.float32)

        payload = (vecs, norms_, compute_distances, message)

        t0 = time.time()
        # Send to all workers (non-blocking sends, then blocking receives)
        reqs = [self.comm.isend(payload, dest=rank, tag=TAG_MESSAGE)
                for rank in range(1, self.num_shards + 1)]
        for req in reqs:
            req.wait()

        all_responses = [self.comm.recv(source=rank, tag=TAG_RESPONSE)
                         for rank in range(1, self.num_shards + 1)]
        rtt = time.time() - t0

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

    def _shutdown_workers(self):
        for rank in range(1, self.num_shards + 1):
            self.comm.send(None, dest=rank, tag=TAG_SHUTDOWN)

    def writeNeighborhood(self, vec_id):
        with open(f"{self.EFS_PATH}/computed.txt", 'a+') as f:
            f.write(f"{vec_id}\n")
        with open(f"{self.EFS_PATH}/neighborhoods.txt", 'a+') as f:
            f.write(f"{self.neighborhoods[vec_id]}\n")

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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True, help="Path to dataset directory")
    parser.add_argument("--num_points", type=int, default=10_000)
    parser.add_argument("--batch",      type=int, default=50)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--cpus",       type=int, default=16,
                        help="BLAS threads per worker (match --cpus-per-task in SLURM)")
    args = parser.parse_args()

    expected = args.num_shards + 1  # coordinator + workers
    if size != expected:
        if rank == 0:
            print(f"Error: need exactly {expected} MPI ranks (1 coordinator + {args.num_shards} workers), got {size}", flush=True)
        MPI.Finalize()
        return

    if rank == 0:
        coordinator = Coordinator(comm, args.data, args.num_points, args.batch, args.num_shards)
        coordinator.computeNeighborhoods()
    else:
        worker_loop(comm, rank, args)


if __name__ == '__main__':
    main()
