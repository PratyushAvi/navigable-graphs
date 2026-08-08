import numpy as np
from utils import *
import argparse
import h5py
from tqdm import tqdm
import pandas as pd
import cupy as cp


def parse_range(spec, n):
    """Parse a half-open "START:END" source range against a dataset of size n.

    Follows Python slice semantics: START defaults to 0, END to n, and END is
    exclusive, so "1:100" yields 1..99. Raises ValueError on anything malformed
    or out of bounds rather than silently clamping — a typo'd range would
    otherwise produce a partial graph that looks complete.
    """
    if spec.count(':') != 1:
        raise ValueError(f"range must be 'START:END', got {spec!r}")
    start_s, end_s = (p.strip() for p in spec.split(':'))
    try:
        start = int(start_s) if start_s else 0
        end   = int(end_s)   if end_s   else n
    except ValueError:
        raise ValueError(f"range bounds must be integers, got {spec!r}")
    if start < 0 or end < 0:
        raise ValueError(f"range bounds must be non-negative, got {spec!r}")
    if end > n:
        raise ValueError(f"range end {end} exceeds dataset size {n}")
    if start >= end:
        raise ValueError(f"range start {start} must be less than end {end}")
    return start, end


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
        help='Sources processed in parallel per batch. '
             'GPU: tune to fit VRAM (~5 * batch_size * n * 4 bytes). '
             'CPU: tune to fit RAM (same formula but RAM is usually larger).'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU (NumPy/BLAS) instead of GPU (CuPy). '
             'Set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS in your job script.'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='alpha-reachability parameter (>= 1). A point p counts as covered by '
             'edge (u, v) when d(v, p) < d(u, p) / alpha, so larger alpha requires '
             'more progress per edge and produces denser neighborhoods. '
             'alpha=1 (the default) is the standard coverage rule. '
             'Euclidean metric only.'
    )
    parser.add_argument(
        '--range',
        type=str,
        default=None,
        help='Restrict which sources to build neighborhoods for, as a Python-style '
             'half-open slice "START:END" (e.g. "1:100" builds sources 1..99). '
             'START defaults to 0 and END to the dataset size, so ":100" and "1:" '
             'both work. Neighbors are still drawn from the whole dataset, so this '
             'splits the work without changing the resulting graph.'
    )
    args = parser.parse_args()
    if args.alpha < 1.0:
        parser.error(f"--alpha must be >= 1, got {args.alpha}")
    DATASET = args.dataset # Get dataset name from argument

    print("Building graph on", DATASET)
    SAVEPATH = "/scratch/pa2439/ANN-Search/navigable_graph_results/new_results"

    DATASETS = dict()
    # Note: pd.read_csv returns a DataFrame; converting to a list of dicts first, then processing.
    dataset_records = pd.read_csv("/scratch/pa2439/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d

    if DATASET not in DATASETS:
        print(f"Error: Dataset '{DATASET}' not found in the loaded metadata.")
        return
    
    if args.metric == 'default':
        metric = DATASETS[DATASET]['metric']
    else:
        metric = args.metric

    # alpha-reachability is only wired through the euclidean prune paths; the
    # angular and jaccard ones still use the plain d(v,p) < d(u,p) rule, so fail
    # loudly rather than silently ignoring the flag.
    if args.alpha != 1.0 and metric != 'euclidean':
        parser.error(f"--alpha is only supported for the euclidean metric, got {metric!r}")

    data = h5py.File(DATASETS[DATASET]['filepath'], 'r')['train']

    use_cpu = args.cpu
    if metric == 'jaccard' or use_cpu:
        dataset = np.asarray(data)
    else:
        dataset = cp.asarray(data)

    n_points = dataset.shape[0]

    # Range is resolved before the output paths are built, since a ranged run
    # writes to its own files and must resume from those, not from the
    # whole-dataset ones.
    if args.range is not None:
        try:
            start, end = parse_range(args.range, n_points)
        except ValueError as e:
            parser.error(str(e))
        all_sources = np.arange(start, end)
        # Tag outputs with the range so concurrent jobs over disjoint ranges
        # don't append to (and interleave within) the same files. The token goes
        # *before* the metric so the metric stays the final '-' segment, which is
        # what the analysis scripts' parse_filename relies on; they strip a
        # "<start>_<end>" token out of the dataset name.
        tag = f"-{start}_{end}"
        print(f"Restricting to sources [{start}:{end}) — {len(all_sources)} of {n_points} points")
    else:
        all_sources = np.arange(n_points)
        tag = ""

    # Each alpha builds a different graph, so the alpha goes in the filename —
    # always, including alpha=1, so every output records what produced it. Like
    # the range tag it goes before the metric, keeping the metric as the final
    # '-' segment for the analysis scripts' parse_filename. '.' would break the
    # '.txt' handling there, so 1.2 is written "alpha1p2".
    tag += "-alpha" + f"{args.alpha:g}".replace('.', 'p')
    print(f"Using alpha-reachability with alpha={args.alpha:g}")

    adj_path      = f"{SAVEPATH}/adj-list-{DATASET}{tag}-{metric}.txt"
    computed_path = f"{SAVEPATH}/{DATASET}{tag}-{metric}-computed.txt"

    # assume no other program is trying to edges simultaneously
    with open(computed_path, 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])

    # FIX: np.random.shuffle returns None, must shuffle the array first
    np.random.shuffle(all_sources)

    sources_to_process = [source for source in all_sources if source not in completed]

    print(f"Writing to {adj_path}")
    if not sources_to_process:
        print("Nothing to do — every source in range is already computed.")
        return

    if metric == 'euclidean':
        # Precompute augmented dataset matrix once; amortised across all batches.
        if use_cpu:
            X_aug  = precomputeAugMatrixCPU(dataset)
            _prune = batchedEuclideanRobustPruneCPU
            print(f"Running on CPU  (batch_size={args.batch_size})")
        else:
            X_aug  = precomputeAugMatrix(dataset)
            _prune = batchedEuclideanRobustPrune
            print(f"Running on GPU  (batch_size={args.batch_size})")

        pbar = tqdm(total=len(sources_to_process))
        for i in range(0, len(sources_to_process), args.batch_size):
            batch = sources_to_process[i : i + args.batch_size]
            neighborhoods = _prune(batch, dataset, X_aug, alpha=args.alpha)

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
