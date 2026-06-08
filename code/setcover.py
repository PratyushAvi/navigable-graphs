import numpy as np
from utils import *
import argparse
import h5py
from tqdm import tqdm
import pandas as pd


def greedySetCover(permutation_matrix, source):
    """
    CPU (NumPy) greedy set cover.

    permutation_matrix[i, j] is the rank of point j in point i's distance-sorted
    order (0 = closest). A point i is "covered via" candidate j when j is closer to
    i than the source is, i.e. permutation_matrix[i, j] < permutation_matrix[i, source].

    Args:
        permutation_matrix: NumPy array (n, n) uint16
        source: int, source vertex index

    Returns:
        list of (neighbor_id, uncov_count) tuples
    """
    n = permutation_matrix.shape[0]
    # per-point threshold: point i is covered via candidate j iff
    # permutation_matrix[i, j] < threshold[i]  (j is closer to i than the source).
    threshold = permutation_matrix[:, source].astype(np.uint16)

    # Work only over the still-uncovered points. The source covers itself.
    # We keep an array of uncovered row indices and shrink it each iteration,
    # so the score reduction gets cheaper as coverage grows — and we never
    # materialise the n x n "sets" matrix.
    uncovered_rows = np.delete(np.arange(n, dtype=np.intp), source)
    edges = []

    while uncovered_rows.size > 0:
        # rows of the permutation matrix for the points still uncovered
        sub = permutation_matrix[uncovered_rows]                 # (u, n) uint16, view-copy
        covers = sub < threshold[uncovered_rows, None]           # (u, n) bool
        scores = covers.sum(axis=0)                              # (n,) int per candidate
        index = int(np.argmax(scores))

        # which currently-uncovered rows does this candidate cover?
        newly = covers[:, index]
        uncovered_rows = uncovered_rows[~newly]
        edges.append((index, int(uncovered_rows.size)))

    return edges


def main():
    parser = argparse.ArgumentParser(description="Compute set-cover edges for a Navigable Graph dataset (CPU).")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="The name of the dataset to process (e.g., 'mnist', 'sift')."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8000,
        help='Rows ranked per batch when building the permutation matrix. '
             'Larger batches parallelise better across cores. Transient RAM per batch '
             'is ~batch_size * n * 8 bytes (float32 block + int64 argsort output).'
    )
    args = parser.parse_args()
    DATASET = args.dataset

    print("Building graph on", DATASET, "(CPU)", flush=True)
    SAVEPATH = "/scratch/pa2439/ANN-Search/navigable_graph_results/new_results"

    DATASETS = dict()
    dataset_records = pd.read_csv("/scratch/pa2439/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d

    if DATASET not in DATASETS:
        print(f"Error: Dataset '{DATASET}' not found in the loaded metadata.")
        return

    metric = 'euclidean'

    # assume no other program is trying to compute edges simultaneously
    with open(f"{SAVEPATH}/{DATASET}-set-cover-{metric}-computed.txt", 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])

    data = h5py.File(DATASETS[DATASET]['filepath'], 'r')['train']

    adj_path      = f"{SAVEPATH}/set-cover-adj-list-{DATASET}-{metric}.txt"
    computed_path = f"{SAVEPATH}/{DATASET}-set-cover-{metric}-computed.txt"

    print("Loading dataset...", flush=True)
    dataset = np.asarray(data, dtype=np.float32)
    n = dataset.shape[0]
    sq_norms = np.einsum('ij,ij->i', dataset, dataset)

    print(f"Building permutation matrix ({n} x {n} uint16, "
          f"{n * n * 2 / 1e9:.1f} GB)...", flush=True)
    permutation_matrix = np.empty((n, n), dtype=np.uint16)
    batch_size = args.batch_size
    row_idx = np.arange(n, dtype=np.uint16)
    for start in tqdm(range(0, n, batch_size), desc="ranking"):
        end = min(start + batch_size, n)
        batch_n = end - start
        # squared euclidean distances for this row block, never the full matrix.
        # float32 is plenty for ordering and halves the transient + speeds the sort.
        dist_block = (sq_norms[start:end, None] + sq_norms[None, :]
                      - 2.0 * (dataset[start:end] @ dataset.T)).astype(np.float32, copy=False)
        np.maximum(dist_block, 0.0, out=dist_block)
        order = np.argsort(dist_block, axis=1)            # nearest -> farthest
        ranks = np.empty((batch_n, n), dtype=np.uint16)
        ranks[np.arange(batch_n)[:, None], order] = row_idx[None, :]
        permutation_matrix[start:end] = ranks
        del dist_block, order, ranks

    for source in tqdm(range(n), desc="set cover"):
        if source in completed:
            continue
        edges = greedySetCover(permutation_matrix, source)
        with open(adj_path, 'a') as adj, open(computed_path, 'a') as comp:
            adj.write(f"{source} {edges}\n")
            comp.write(f"{source}\n")

    print(f"Done with {DATASET}")


if __name__ == "__main__":
    main()
