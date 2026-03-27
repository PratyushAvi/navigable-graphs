"""
Split vectors.npy and sq_norms.npy into per-shard files.
Run once as a preprocessing step before distributed_robust_prune.py.

Usage:
    python split_dataset.py --data /path/to/dataset --num_shards 8
"""
import argparse
import os
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,          help="Path to dataset directory")
    parser.add_argument("--num_shards", type=int, required=True, help="Number of shards to split into")
    args = parser.parse_args()

    print("Opening memory maps...", flush=True)
    vectors = np.load(f"{args.data}/vectors.npy",   mmap_mode='r')
    norms   = np.load(f"{args.data}/sq_norms.npy",  mmap_mode='r')
    total   = vectors.shape[0]
    print(f"Total vectors: {total:,}  shape: {vectors.shape}  dtype: {vectors.dtype}", flush=True)

    shard_size = total // args.num_shards
    t_total    = time.time()

    for i in range(args.num_shards):
        start = i * shard_size
        end   = total if i == args.num_shards - 1 else (i + 1) * shard_size
        n     = end - start

        vec_path  = f"{args.data}/vectors_shard_{i}.npy"
        norm_path = f"{args.data}/sq_norms_shard_{i}.npy"

        if os.path.exists(vec_path) and os.path.exists(norm_path):
            print(f"[Shard {i}] already exists, skipping.", flush=True)
            continue

        print(f"[Shard {i}] [{start:,}, {end:,})  n={n:,}", flush=True)

        t0 = time.time()
        np.save(vec_path, vectors[start:end])
        print(f"[Shard {i}] vectors saved: {time.time()-t0:.1f}s", flush=True)

        t1 = time.time()
        np.save(norm_path, norms[start:end])
        print(f"[Shard {i}] norms saved:   {time.time()-t1:.1f}s  total: {time.time()-t0:.1f}s", flush=True)

    boundaries = np.array([
        [i * shard_size, total if i == args.num_shards - 1 else (i + 1) * shard_size]
        for i in range(args.num_shards)
    ], dtype=np.int64)
    np.save(f"{args.data}/shard_boundaries.npy", boundaries)
    print(f"Shard boundaries saved to shard_boundaries.npy", flush=True)
    print(f"\nAll shards written in {time.time()-t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
