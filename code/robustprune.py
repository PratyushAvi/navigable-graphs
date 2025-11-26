import numpy as np
import cupy as cp
import h5py
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import rankdata
import psutil
from utils import *

def main():
    DATAPATH = "/scratch/pa2439/Projects/ANN-Search/datasets/"
    DATASETS = ["fashion-mnist-784-euclidean", "mnist-784-euclidean", "sift-128-euclidean"]
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/code/results/newrobustprune/"
    # DATASET = "fashion-mnist-784-euclidean"
    

    for DATASET in DATASETS[2:]:
        print(f"Building graphs on {DATASET}")
        data = h5py.File(DATAPATH + DATASET + ".hdf5")

        dataset = cp.asarray(data['train'][:])
        n = dataset.shape[0]

        robustPruneGraph = memBuildRobustPruneGraph(dataset)

        saveGraph(n, robustPruneGraph, f"{SAVEPATH}adjlist-{DATASET}-{n}-vertices-robust-prune.dat")

    '''
    for DATASET in DATASETS[1:2]:
        print(f"Building graphs on {DATASET}")
        data = h5py.File(DATAPATH + DATASET + ".hdf5")
        # ratios = [0.1, 0.2, 0.4, 0.6, 0.8, 1]
        ratios = [0.6, 0.8, 1]
        
        for r in ratios:
            n = int(data['train'].shape[0] * r)

            # Load dataset to CPU first
            # dataset = data['train'][:n]
            dataset = data['train'][np.sort(np.random.choice(data['train'].shape[0], n, replace=False))]

            # Transfer to GPU and compute pairwise distances on GPU in chunks

            chunk_size = auto_chunk_size(dataset.shape)  # Adjust based on your memory

            permutation_matrix = cp.empty((n, n), dtype=cp.uint16 if np.log2(n) <= 16 else cp.uint32)

            for i in range(0, n, chunk_size):
                end_i = min(i + chunk_size, n)
                
                # Compute pairwise distances for this chunk (on CPU)
                chunk_distances = cdist(dataset[i:end_i], dataset, metric='euclidean')
                
                # Set diagonal to -1 for rows that correspond to diagonal elements
                for j in range(end_i - i):
                    if i + j < n:
                        chunk_distances[j, i + j] = -1
                
                # Compute ranks and transfer to GPU
                chunk_ranks = rankdata(chunk_distances, method='ordinal', axis=1)
                print(f"Chunk shape: {chunk_ranks.shape}")
                permutation_matrix[i:end_i] = cp.asarray(chunk_ranks, dtype=cp.uint16 if np.log2(n) <= 16 else cp.uint32)

            
            print("Permutation matrix shape:", permutation_matrix.shape)

            # Build robust prune graph on GPU
            robustPruneGraph = buildRobustPruneGraph(permutation_matrix)

            # store built graph
            saveGraph(n, robustPruneGraph, f"{SAVEPATH}adjlist-{DATASET}-{n}-vertices-robust-prune.dat")
    '''

if __name__ == '__main__':
    main()
