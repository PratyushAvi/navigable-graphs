import numpy as np
from tqdm import tqdm
from time import time
from scipy.spatial.distance import cdist

def main():
    DATASET = "spacev1b" # Get dataset name from argument
    METRIC = 'euclidean'

    print("Building graph on", DATASET, flush=True)
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    BINARY_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_int.mmap"
    dataset_shape = np.load("/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_shape.npy")

    X = np.memmap(BINARY_FILE, mode='r', dtype='int8', shape=dataset_shape)
    print("Loaded vectors:", X.shape, flush=True)


    source = 0

    SHARD_SIZE = 10 ** 7

    N = []

    for i in range(X.shape[0] // SHARD_SIZE + 1):
        start = time()
        print("\nLoading shard into RAM...", flush=True)
        shard = X[SHARD_SIZE * i: SHARD_SIZE * (i+1)]
        print(f"Shard shape: {shard.shape}\nComputing out-neighborhood", flush=True)
        N_i = [p + SHARD_SIZE * i for p in shardedRobustPrune(X[source:source+1], shard)]
        N.append(N_i)
        print(f"Took {time() - start} seconds...")

    N = np.sort(np.array(sum(N, [])))
    edges = np.sort(N[shardedRobustPrune(X[source:source+1], X[N])])
    print('out-neighborhood size:', len(edges))
    print()
    print(edges)


def shardedRobustPrune(p, X):
    n = X.shape[0]
    print("Starting!", flush=True)
    # dist_from_source = -2 * (X @ p) + (p @ p)
    dist_from_source = cdist(p, X, metric='sqeuclidean')
    print("computed pairwise distances...", flush=True)

    edges = []

    active = np.ones(X.shape[0], dtype=np.uint8)
    
    while np.any(active):
        print(np.sum(active), "uncovered...\r", flush=True)
        masked_dist = np.where(active, dist_from_source, np.inf)
        waypoint = np.argmin(masked_dist).item()
        
        edges.append(waypoint)
        active[waypoint] = False

        # dist_from_waypoint = -2 * (X @ X[waypoint]) + (X[waypoint] @ X[waypoint])
        dist_from_waypoint = cdist(X[waypoint:waypoint+1], X, metric='sqeuclidean')
        active[dist_from_waypoint < dist_from_source] = 0

    return edges

if __name__ == '__main__':
    main()
