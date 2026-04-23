import numpy as np
import cupy as cp
from utils import *
import argparse
import h5py
from tqdm import tqdm
import pandas as pd
from cupyx.scipy.spatial.distance import cdist


def main():
    # 1. Setup ArgParse (MINIMAL CHANGE)
    parser = argparse.ArgumentParser(description="Compute edges for a Navigable Graph dataset.")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="The name of the dataset to process (e.g., 'mnist', 'sift')."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Sources processed in parallel per batch. '
             'GPU: tune to fit VRAM (~5 * batch_size * n * 4 bytes). '
             'CPU: tune to fit RAM (same formula but RAM is usually larger).'
    )
    args = parser.parse_args()
    DATASET = args.dataset # Get dataset name from argument
    use_cpu = True

    print("Building graph on", DATASET, flush=True)
    SAVEPATH = "/scratch/pa2439/ANN-Search/navigable_graph_results/new_results"

    DATASETS = dict()
    # Note: pd.read_csv returns a DataFrame; converting to a list of dicts first, then processing.
    dataset_records = pd.read_csv("/scratch/pa2439/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d

    if DATASET not in DATASETS:
        print(f"Error: Dataset '{DATASET}' not found in the loaded metadata.")
        return
    
    metric = 'euclidean'

    # assume no other program is trying to edges simultaneously
    with open(f"{SAVEPATH}/{DATASET}-set-cover-{metric}-computed.txt", 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])

    data = h5py.File(DATASETS[DATASET]['filepath'], 'r')['train']

    # all_sources = np.arange(dataset.shape[0])
    # np.random.shuffle(all_sources)

    # sources_to_process = [source for source in all_sources if source not in completed]

    adj_path      = f"{SAVEPATH}/set-cover-adj-list-{DATASET}-{metric}.txt"
    computed_path = f"{SAVEPATH}/{DATASET}-set-cover-{metric}-computed.txt"

    # Precompute augmented dataset matrix once; amortised across all batches.

    print("Computing pairwise distances...", flush=True)
    dataset_cp = cp.asarray(data, dtype=cp.float32)
    sq_norms = cp.einsum('ij,ij->i', dataset_cp, dataset_cp)
    dist_matrix = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (dataset_cp @ dataset_cp.T)
    cp.maximum(dist_matrix, 0.0, out=dist_matrix)

    print("Building permutation matrix...", flush=True)
    permutation_matrix = cp.argsort(cp.argsort(dist_matrix, axis=1), axis=1).astype(cp.uint16)

    for source in tqdm(range(len(dataset_cp))):
        # Build sets for this source on GPU
        # print(f"{source} building sets", end='\r')
        sets = buildSetsOfSource(permutation_matrix, source)
        # print(f"{source} done building", end='\r')
        # Run greedy set cover on GPU
        edges = greedySetCover(sets, permutation_matrix, source)
        with open(adj_path, 'a') as adj, open(computed_path, 'a') as comp:
            adj.write(f"{source} {edges}\n")
            comp.write(f"{source}\n")
        
    print(f"Done with {DATASET}")

if __name__ == "__main__":
    main()
