import numpy as np
from utils import *
import argparse
import h5py
from tqdm import tqdm
import pandas as pd
import cupy as cp


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
        '--metric',
        type=str,
        default='default',
        help='Distance metric (e.g. euclidean, angular)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Number of sources processed in parallel per GPU batch (euclidean only). '
             'Each batch holds 2 * batch_size * n float32 values on the GPU; tune to fit VRAM.'
    )
    args = parser.parse_args()
    DATASET = args.dataset # Get dataset name from argument

    print("Building graph on", DATASET)
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"

    DATASETS = dict()
    # Note: pd.read_csv returns a DataFrame; converting to a list of dicts first, then processing.
    dataset_records = pd.read_csv("/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d

    if DATASET not in DATASETS:
        print(f"Error: Dataset '{DATASET}' not found in the loaded metadata.")
        return
    
    if args.metric == 'default':
        metric = DATASETS[DATASET]['metric']
    else:
        metric = args.metric

    # assume no other program is trying to edges simultaneously
    with open(f"{SAVEPATH}/{DATASET}-{metric}-computed.txt", 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])

    data = h5py.File(DATASETS[DATASET]['filepath'], 'r')['train']
    
    if metric != 'jaccard':
        dataset = cp.asarray(data)
    else:
        dataset = np.asarray(data)

    # FIX: np.random.shuffle returns None, must shuffle the array first
    all_sources = np.arange(dataset.shape[0])
    np.random.shuffle(all_sources)

    sources_to_process = [source for source in all_sources if source not in completed]
    


    adj_path      = f"{SAVEPATH}/adj-list-{DATASET}-{metric}.txt"
    computed_path = f"{SAVEPATH}/{DATASET}-{metric}-computed.txt"

    if metric == 'euclidean':
        # Precompute augmented dataset matrix once; amortised across all batches.
        X_aug = precomputeAugMatrix(dataset)

        pbar = tqdm(total=len(sources_to_process))
        for i in range(0, len(sources_to_process), args.batch_size):
            batch = sources_to_process[i : i + args.batch_size]
            neighborhoods = batchedEuclideanRobustPrune(batch, dataset, X_aug)

            # Write entire batch at once; a crash loses at most batch_size sources of work.
            with open(adj_path, 'a') as adj, open(computed_path, 'a') as comp:
                for j, source in enumerate(batch):
                    adj.write(f"{source} {neighborhoods[j]}\n")
                    comp.write(f"{source}\n")

            pbar.update(len(batch))
        pbar.close()

    elif metric == 'angular':
        for source in tqdm(sources_to_process):
            edges = angularRobustPrune(source, dataset)
            with open(adj_path, 'a') as adj:
                adj.write(','.join([str(e) for e in edges]) + '\n')
            with open(computed_path, 'a') as f:
                f.write(f"{source}\n")

    elif metric == 'jaccard':
        for source in tqdm(sources_to_process):
            edges = jaccardRobustPrune(source, dataset)
            with open(adj_path, 'a') as adj:
                adj.write(','.join([str(e) for e in edges]) + '\n')
            with open(computed_path, 'a') as f:
                f.write(f"{source}\n")

    else:
        print('dunno')
        return
        
    print(f"Done with {DATASET}")

if __name__ == "__main__":
    main()
