import cupy as cp
import numpy as np
from tqdm import tqdm
from time import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Build navigable graph on dataset')
    parser.add_argument('--shard-size', type=int, default=10**7,
                        help='Number of vectors per shard (default: 10000000)')
    args = parser.parse_args()
    
    DATASET = "spacev1b"
    METRIC = 'euclidean'
    print("Building graph on", DATASET, flush=True)
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    BINARY_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_int.mmap"
    dataset_shape = np.load("/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_shape.npy")
    X = np.memmap(BINARY_FILE, mode='r', dtype='int8', shape=dataset_shape)
    print("Loaded vectors:", X.shape, flush=True)
    
    SHARD_SIZE = args.shard_size
    print(f"Using shard size: {SHARD_SIZE}", flush=True)
    
    # Load completed sources
    with open(f"{SAVEPATH}/{DATASET}-{METRIC}-computed.txt", 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])
    
    # Generate all sources
    all_sources = np.arange(X.shape[0], dtype=np.uint32)
    np.random.shuffle(all_sources)
    
    # Process each source
    for source in all_sources:  # Process first source (change [:1] to process more)
        if source in completed:
            print(f"Source {source} already completed, skipping", flush=True)
            continue
            
        print(f"\nProcessing source {source}", flush=True)
        start_total = time()
        
        # Step 1: Process all shards to find initial candidates
        N = []
        for i in tqdm(range(X.shape[0] // SHARD_SIZE + 1), desc="Processing shards"):
            shard = X[SHARD_SIZE * i: SHARD_SIZE * (i+1)]
            if shard.shape[0] == 0:  # Skip empty shards
                continue
            N_i = [p + SHARD_SIZE * i for p in shardedRobustPrune(X[source:source+1], shard)]
            N.append(N_i)
        
        # Step 2: Combine all candidates and do final pruning
        N = np.sort(np.array(sum(N, [])))
        print(f"Initial candidates: {len(N)}", flush=True)
        
        edges = [source] + sorted([N[idx] for idx in shardedRobustPrune(X[source:source+1], X[N])])
        
        print(f"Out-neighborhood size: {len(edges) - 1}", flush=True)
        print(f"Total time: {time() - start_total:.2f} seconds", flush=True)
        
        # Save results
        with open(f"{SAVEPATH}/adj-list-{DATASET}-{METRIC}.txt", 'a') as adj:
            adj.write(','.join([str(e) for e in edges]) + '\n')
        
        with open(f"{SAVEPATH}/{DATASET}-{METRIC}-computed.txt", 'a') as f:
            f.write(f"{source}\n")
    
    print(f"Done with {DATASET}")

def shardedRobustPrune(p, X):
    """
    Robust prune algorithm on GPU.
    Returns indices of edges (waypoints) that cover all points in X.
    """
    n = X.shape[0]
    
    # Transfer to GPU
    p_gpu = cp.asarray(p, dtype=cp.float32)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    
    # Compute squared euclidean distance on GPU
    dist_from_source = cp.sum((p_gpu - X_gpu) ** 2, axis=1).flatten()
    
    edges = []
    active = cp.ones(X_gpu.shape[0], dtype=cp.uint8)
    
    while cp.any(active):
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()
        edges.append(waypoint)
        active[waypoint] = False
   
        # Compute distance from waypoint on GPU
        dist_from_waypoint = cp.sum((X_gpu[waypoint:waypoint+1] - X_gpu) ** 2, axis=1).flatten()
        active = cp.where(dist_from_waypoint < dist_from_source, 0, active)
   
    return edges

if __name__ == '__main__':
    main()
