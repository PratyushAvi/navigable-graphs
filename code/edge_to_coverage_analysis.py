import ast
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse


COLUMNS = ['dataset', 'metric', 'method', 'dimensions', 'sources', 'total points',
           'edges', 'mean coverage', 'median coverage',
           'min coverage', 'max coverage']


def main():
    parser = argparse.ArgumentParser(
        description='Compute average coverage achieved for a given number of edges')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to process. If omitted, processes all datasets found. '
                             'Required when --adj-list is given (used as the dataset name in output).')
    parser.add_argument('--adj-list', type=str, default=None,
                        help='Path to a single adjacency-list file to process. When set, skips scanning '
                             'SAVEPATH for robust-prune/set-cover files; requires --dataset.')
    parser.add_argument('--computed', type=str, default=None,
                        help='Path to the computed-sources file for --adj-list (the source ids, one per '
                             'line). Provide for robust-prune outputs whose adj-list lines are bare '
                             'neighborhoods; omit for "<source_id> <neighborhood>" lines.')
    parser.add_argument('--metric', type=str, default='',
                        help='Metric label, recorded in the output only; has no effect on computation '
                             '(default: empty).')
    parser.add_argument('--method', type=str, default='robust-prune', choices=['robust-prune', 'set-cover'],
                        help='Method label for the output when using --adj-list (default: robust-prune).')
    parser.add_argument('--min-edges', type=int, default=1,
                        help='Minimum number of edges to report coverage for (default: 1)')
    parser.add_argument('--max-edges', type=int, default=64,
                        help='Maximum number of edges to report coverage for (default: 64)')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Step size between edge counts (default: 1)')
    parser.add_argument('--total-points', type=int, required=True,
                        help='Size of the full dataset that coverage is measured against '
                             '(the universe uncov counts range over, e.g. 1000000000 for bigann)')
    parser.add_argument('--dimensions', type=int, default=-1,
                        help='Vector dimensionality, recorded in the output for reference (default: -1)')
    args = parser.parse_args()

    # Build sorted list of edge counts to report coverage at.
    edge_counts = list(range(args.min_edges, args.max_edges + 1, args.step_size))
    n_e = len(edge_counts)
    max_e = edge_counts[-1]
    print(f"Edge counts ({n_e}): {edge_counts[0]} to {edge_counts[-1]} step {args.step_size}")

    SAVEPATH = "/scratch/pa2439/ANN-Search/navigable_graph_results/new_results"
    STATS_DIR = "/scratch/pa2439/ANN-Search/navigable_graph_results/edge_to_coverage"

    if args.adj_list is not None:
        # Explicit single-file mode: caller supplies the file and its labels.
        if args.dataset is None:
            parser.error("--adj-list requires --dataset")
        if not os.path.exists(args.adj_list):
            parser.error(f"--adj-list path does not exist: {args.adj_list}")
        if args.computed is not None and not os.path.exists(args.computed):
            parser.error(f"--computed path does not exist: {args.computed}")

        labels = {args.adj_list: (args.dataset, args.metric)}

        def parse_filename(file, method):
            return labels[file]

        files_to_process = [(args.adj_list, args.method)]
        print(f"Processing single adjacency list: {args.adj_list} "
              f"[{args.dataset}-{args.metric}, {args.method}]")
    else:
        adjLists = (
            [(f, 'robust-prune') for f in glob.glob(f"{SAVEPATH}/adj-list-*.txt")] +
            [(f, 'set-cover')    for f in glob.glob(f"{SAVEPATH}/set-cover-adj-list-*.txt")]
        )

        def parse_filename(file, method):
            stem = os.path.basename(file).replace(".txt", "")
            parts = stem.split("-")[4:] if method == 'set-cover' else stem.split("-")[2:]
            return "-".join(parts[:-1]), parts[-1]

        files_to_process = []
        for file, method in adjLists:
            dataset_name, metric = parse_filename(file, method)
            if args.dataset is not None and dataset_name != args.dataset:
                continue
            files_to_process.append((file, method))

        print(f"Found {len(files_to_process)} files to process out of {len(adjLists)} total")

    if not files_to_process:
        print("Nothing to do.")
        return

    all_new_rows = []

    for file, method in tqdm(files_to_process, desc="Processing adjacency lists"):
        dataset_name, metric = parse_filename(file, method)
        n_nodes = args.total_points
        print(f"\nProcessing {dataset_name}-{metric} [{method}] ({n_nodes} nodes)")

        # cov_sum[e_idx] = sum over sources of coverage (%) achieved using the
        # first edge_counts[e_idx] edges of that source's neighborhood.
        # We also track min/max per edge count for reporting.
        cov_sum = np.zeros(n_e, dtype=np.float64)
        cov_min = np.full(n_e, np.inf, dtype=np.float64)
        cov_max = np.full(n_e, -np.inf, dtype=np.float64)
        cov_all = []  # per-source coverage vectors, for the median

        counter = 0

        # distributed_robust_prune writes the source ids to a separate
        # computed-sources file and bare neighborhoods (no source prefix) to the
        # adj-list. set-cover / legacy outputs instead prefix each adj-list line
        # with "<source_id> ". In single-file mode the caller passes --computed
        # explicitly; in scan mode we look for {dataset}-{metric}-computed.txt.
        if args.adj_list is not None:
            computed_txt = args.computed
        elif method == 'robust-prune':
            computed_txt = f"{SAVEPATH}/{dataset_name}-{metric}-computed.txt"
            computed_txt = computed_txt if os.path.exists(computed_txt) else None
        else:
            computed_txt = None

        computed_sources = None
        if computed_txt is not None:
            with open(computed_txt, 'r') as cf:
                computed_sources = [int(p.strip()) for p in cf if p.strip()]
            print(f"  {dataset_name}-{metric}: {len(computed_sources)} computed sources")

        with open(file, 'r') as f:
            first_line = f.readline().strip()
            if '[' not in first_line:
                print(f"  Skipping {file}: legacy format without uncov data")
                continue
            f.seek(0)

            for line in tqdm(f, desc="Sources", leave=False):
                line = line.strip()
                if not line:
                    continue

                if computed_sources is not None:
                    if counter >= len(computed_sources):
                        break
                    neighborhood = ast.literal_eval(line)
                else:
                    space = line.index(' ')
                    neighborhood = ast.literal_eval(line[space + 1:])

                counter += 1

                # coverage after including the first k edges is
                # (n_nodes - uncov_after_k) / n_nodes * 100, where uncov_after_k
                # is the uncov value recorded on the k-th edge (edges are stored
                # in insertion order, so uncov is non-increasing along the list).
                #
                # If a source has degree d < k, using k edges is the same as
                # using all d edges: its coverage saturates at its final value
                # (100% when the neighborhood fully covers the universe). So a
                # source with degree d contributes its degree-d coverage to every
                # edge count >= d.
                deg = len(neighborhood)
                # coverage at each of this source's own edge indices (1..deg)
                # cov_by_edge[j] = coverage using first (j+1) edges
                cov_by_edge = np.empty(deg, dtype=np.float64)
                for j, (neighbor, uncov) in enumerate(neighborhood):
                    cov_by_edge[j] = (n_nodes - uncov) / n_nodes * 100.0

                cov_src = np.empty(n_e, dtype=np.float64)
                for e_idx, k in enumerate(edge_counts):
                    if deg == 0:
                        cov_src[e_idx] = 0.0
                    elif k >= deg:
                        cov_src[e_idx] = cov_by_edge[deg - 1]
                    else:
                        cov_src[e_idx] = cov_by_edge[k - 1]

                cov_sum += cov_src
                cov_min = np.minimum(cov_min, cov_src)
                cov_max = np.maximum(cov_max, cov_src)
                cov_all.append(cov_src)

        if counter == 0:
            print(f"  No sources found — skipping")
            continue

        print(f"  Processed {counter} sources")

        cov_matrix = np.array(cov_all, dtype=np.float64)  # (counter, n_e)
        cov_mean = cov_sum / counter
        cov_median = np.median(cov_matrix, axis=0)

        for e_idx, k in enumerate(edge_counts):
            all_new_rows.append([
                dataset_name,
                metric,
                method,
                args.dimensions,
                counter,
                n_nodes,
                k,
                float(np.round(cov_mean[e_idx], 4)),
                float(np.round(cov_median[e_idx], 4)),
                float(np.round(cov_min[e_idx], 4)),
                float(np.round(cov_max[e_idx], 4)),
            ])

        print(f"  Computed coverage for {n_e} edge counts")

    # --- Save to CSV (one file per dataset) ---
    new_df = pd.DataFrame(all_new_rows, columns=COLUMNS)

    os.makedirs(STATS_DIR, exist_ok=True)
    for dataset_name, grp in new_df.groupby('dataset'):
        stats_file = os.path.join(STATS_DIR, f"edge_to_coverage_stats_{dataset_name}.csv")
        grp.to_csv(stats_file, index=False)
        print(f"\nSaved {len(grp)} rows → {stats_file}")


if __name__ == '__main__':
    main()
