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

    # assume no other program is trying to edges simultaneously
    with open(f"{SAVEPATH}/{DATASET}-{DATASETS[DATASET]['metric']}-computed.txt", 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])

    data = h5py.File(DATASETS[DATASET]['filepath'], 'r')['train']
    
    if DATASETS[DATASET]['metric'] != 'jaccard':
        dataset = cp.asarray(data)
    else:
        dataset = np.asarray(data)

    # FIX: np.random.shuffle returns None, must shuffle the array first
    all_sources = np.arange(dataset.shape[0])
    np.random.shuffle(all_sources)

    sources_to_process = [source for source in all_sources if source not in completed]
    


    # pick specific function
    if DATASETS[DATASET]['metric'] == 'euclidean':
        buildGraph = memEfficientRobustPrune
    elif DATASETS[DATASET]['metric'] == 'angular':
        buildGraph = angularRobustPrune
    elif DATASETS[DATASET]['metric'] == 'jaccard':
        buildGraph = jaccardRobustPrune
    else:
        print('dunno')
        return

    for source in tqdm(sources_to_process):
        if source not in completed:

            edges = buildGraph(source, dataset)

            with open(f"{SAVEPATH}/adj-list-{DATASET}-{DATASETS[DATASET]['metric']}.txt", 'a') as adj:
                # FIX: Added closing parenthesis and newline character
                adj.write(','.join([str(e) for e in edges]) + '\n')
            
            with open(f"{SAVEPATH}/{DATASET}-{DATASETS[DATASET]['metric']}-computed.txt", 'a') as f:
                f.write(f"{source}\n")
        
    print(f"Done with {DATASET}")

if __name__ == "__main__":
    main()
