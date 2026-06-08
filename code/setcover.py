import numpy as np
from utils import *
import argparse
import sys
import h5py
from tqdm import tqdm
import pandas as pd

# When stdout is a real terminal we use animated tqdm bars. When it is redirected
# to a file (e.g. a SLURM log viewed with `tail -f`), animated bars spam the log
# with carriage-return redraws, so we switch to plain newline-terminated progress
# lines emitted on an interval.
IS_TTY = sys.stdout.isatty()


def greedySetCover(permutation_matrix, source, inner_bar=False):
    """
    CPU (NumPy) greedy set cover.

    permutation_matrix[i, j] is the rank of point j in point i's distance-sorted
    order (0 = closest). A point i is "covered via" candidate j when j is closer to
    i than the source is, i.e. permutation_matrix[i, j] < permutation_matrix[i, source].

    Args:
        permutation_matrix: NumPy array (n, n) uint16
        source: int, source vertex index
        inner_bar: if True (interactive terminal only), show a per-source tqdm bar
                   tracking points covered. Disabled by default so it does not spam
                   a redirected log (SLURM / `tail -f`).

    Returns:
        list of (neighbor_id, uncov_count) tuples
    """
    n = permutation_matrix.shape[0]
    # per-point threshold: point i is covered via candidate j iff
    # permutation_matrix[i, j] < threshold[i]  (j is closer to i than the source).
    threshold = permutation_matrix[:, source].astype(np.uint16)

    # Build the boolean "sets" matrix ONCE for this source:
    #   covers[i, j] = True  iff candidate j covers point i.
    # n x n bool = ~12.8 GB at n=113k. We then reduce over only the still-uncovered
    # rows each iteration using a boolean mask — no per-iteration gather/copy, so
    # peak memory is resident perm matrix (25.7 GB) + this bool matrix (12.8 GB).
    covers = permutation_matrix < threshold[:, None]             # (n, n) bool, built once

    uncovered = np.ones(n, dtype=np.bool_)
    uncovered[source] = False                                    # source covers itself
    n_uncovered = n - 1
    edges = []

    bar = (tqdm(total=n_uncovered, desc=f"  cover src {source}",
                unit="pt", leave=False, position=1) if inner_bar else None)

    while n_uncovered > 0:
        # score candidates by how many still-uncovered points they cover.
        # restrict the reduction to uncovered rows (no copy of the matrix).
        scores = covers[uncovered].sum(axis=0)                   # (n,) int per candidate
        index = int(np.argmax(scores))

        newly = covers[:, index] & uncovered                    # uncovered rows j covers
        newly_count = int(newly.sum())
        uncovered &= ~newly
        n_uncovered -= newly_count
        edges.append((index, n_uncovered))

        if bar is not None:
            bar.update(newly_count)
            bar.set_postfix(edges=len(edges), uncov=n_uncovered, refresh=False)

    if bar is not None:
        bar.close()
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

    import time
    total_edges = 0
    processed = 0
    remaining = [s for s in range(n) if s not in completed]
    print(f"Set cover: {len(remaining)} of {n} sources to process "
          f"({len(completed)} already done).", flush=True)

    # Interactive terminal: animated tqdm bar. Redirected log (SLURM / tail -f):
    # plain newline-terminated lines so the log stays readable.
    # Log EVERY source for the first few (so you can see it is alive and how slow
    # one source is), then thin out to ~200 lines total for the whole run.
    n_rem = len(remaining)
    log_every = max(1, n_rem // 200)
    t_start = time.time()
    outer = tqdm(remaining, desc="set cover", unit="src", disable=not IS_TTY)

    for source in outer:
        t_src = time.time()
        edges = greedySetCover(permutation_matrix, source, inner_bar=IS_TTY)
        src_secs = time.time() - t_src
        total_edges += len(edges)
        processed += 1
        with open(adj_path, 'a') as adj, open(computed_path, 'a') as comp:
            adj.write(f"{source} {edges}\n")
            comp.write(f"{source}\n")

        if IS_TTY:
            outer.set_postfix(deg=len(edges),
                              avg_deg=f"{total_edges / processed:.1f}",
                              total_edges=total_edges, refresh=False)
            continue

        # log file: chatty at the start, then periodic
        if processed <= 5 or processed % log_every == 0 or processed == n_rem:
            elapsed = time.time() - t_start
            rate = processed / elapsed
            eta = (n_rem - processed) / rate if rate > 0 else 0.0
            print(f"[set cover] {processed}/{n_rem} | src={source} deg={len(edges)} "
                  f"| {src_secs:.1f}s/src | avg_deg={total_edges / processed:.1f} "
                  f"| {rate:.3f} src/s | ETA {eta / 3600:.1f}h",
                  flush=True)

    print(f"Done with {DATASET}: {processed} sources, {total_edges} edges total.", flush=True)


if __name__ == "__main__":
    main()
