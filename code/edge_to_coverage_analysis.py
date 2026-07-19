import ast
import glob
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os
import argparse

# "<start>_<end>" source-range token written by simulrun.py --range
RANGE_TOKEN = re.compile(r"\d+_\d+")


COLUMNS = ['dataset', 'metric', 'method', 'dimensions', 'sources', 'total points',
           'edges', 'mean points covered', 'median points covered',
           'min points covered', 'max points covered',
           'sources below 99.5% coverage', 'sources below 100% coverage']


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
    parser.add_argument('--max-edges', type=int, default=None,
                        help='Maximum number of edges to report coverage for. If omitted, '
                             'defaults to the largest out-degree seen in the data (per dataset).')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Step size between edge counts (default: 1)')
    parser.add_argument('--total-points', type=int, required=True,
                        help='Size of the full dataset that coverage is measured against '
                             '(the universe uncov counts range over, e.g. 1000000000 for bigann)')
    parser.add_argument('--dimensions', type=int, default=-1,
                        help='Vector dimensionality, recorded in the output for reference (default: -1)')
    args = parser.parse_args()

    # Edge counts to report coverage at are built per file below: when
    # --max-edges is omitted we cap at that file's largest out-degree, so the
    # curve runs exactly as far as the data supports.
    if args.max_edges is not None:
        print(f"Edge counts: {args.min_edges} to {args.max_edges} step {args.step_size}")
    else:
        print(f"Edge counts: {args.min_edges} to <max out-degree per dataset> "
              f"step {args.step_size}")

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
            metric, name_parts = parts[-1], parts[:-1]
            # simulrun.py --range tags partial runs as "<dataset>-<start>_<end>",
            # e.g. adj-list-sift-1_100-euclidean.txt. Drop that token so every
            # shard of a dataset aggregates under the one dataset name.
            if len(name_parts) > 1 and RANGE_TOKEN.fullmatch(name_parts[-1]):
                name_parts = name_parts[:-1]
            return "-".join(name_parts), metric

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

        # Per source, store its points-covered-by-edge curve (length = its degree).
        # We can't fix the reported edge counts until we've seen every source,
        # since --max-edges may default to the largest out-degree in the file.
        per_source_cov = []  # list of 1-D float arrays, cov_by_edge[j] for j<deg
        max_deg = 0

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

                # Each adj-list tuple is (edge, uncov_after_edge): the uncov count
                # after that edge takes effect. So points covered after including
                # the first k edges is n_nodes - uncov recorded on the k-th edge.
                # (Older files predating this convention store the pre-edge uncov;
                # run fix_adj_list_uncov.py to migrate them first.) Edges are in
                # insertion order, so uncov is non-increasing along the list. Kept
                # as exact integers to avoid rounding at large n_nodes.
                # cov_by_edge[j] = points covered using the first (j+1) edges.
                deg = len(neighborhood)
                cov_by_edge = np.empty(deg, dtype=np.int64)
                for j, (neighbor, uncov) in enumerate(neighborhood):
                    cov_by_edge[j] = n_nodes - uncov

                per_source_cov.append(cov_by_edge)
                if deg > max_deg:
                    max_deg = deg

        if counter == 0:
            print(f"  No sources found — skipping")
            continue

        print(f"  Processed {counter} sources (max out-degree {max_deg})")

        # Now fix the edge counts to report. Default the upper bound to the
        # largest out-degree seen; beyond that no source has more edges.
        max_edges = args.max_edges if args.max_edges is not None else max_deg
        edge_counts = list(range(args.min_edges, max_edges + 1, args.step_size))
        if not edge_counts:
            print(f"  No edge counts in [{args.min_edges}, {max_edges}] — skipping")
            continue
        n_e = len(edge_counts)

        # Build the (counter, n_e) points-covered matrix. If a source has degree
        # d < k, using k edges is the same as using all d edges: its coverage
        # saturates at its final value (n_nodes when it fully covers). So a source
        # of degree d contributes its degree-d points covered to every edge count >= d.
        edge_idx = np.array(edge_counts) - 1  # 0-based positions to sample
        cov_matrix = np.empty((counter, n_e), dtype=np.int64)
        for i, cov_by_edge in enumerate(per_source_cov):
            deg = len(cov_by_edge)
            if deg == 0:
                cov_matrix[i, :] = 0
            else:
                # clamp requested edge index to this source's last edge (deg-1)
                cov_matrix[i, :] = cov_by_edge[np.minimum(edge_idx, deg - 1)]

        cov_mean = cov_matrix.mean(axis=0)            # fractional: average of counts
        cov_median = np.median(cov_matrix, axis=0)    # may be x.5 for even counts
        cov_min = cov_matrix.min(axis=0)
        cov_max = cov_matrix.max(axis=0)

        # Per edge count, how many sources have NOT yet reached the given coverage
        # level (i.e. covered fewer points than the threshold). 99.5% uses a floor
        # so a source exactly at the boundary counts as covered; 100% is exact
        # (covered < n_nodes means still incomplete).
        thresh_995 = 0.995 * n_nodes
        n_below_995 = (cov_matrix < thresh_995).sum(axis=0)
        n_below_100 = (cov_matrix < n_nodes).sum(axis=0)

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
                float(cov_median[e_idx]),
                int(cov_min[e_idx]),
                int(cov_max[e_idx]),
                int(n_below_995[e_idx]),
                int(n_below_100[e_idx]),
            ])

        print(f"  Computed points covered for {n_e} edge counts")

    # --- Save to CSV (one file per dataset) ---
    new_df = pd.DataFrame(all_new_rows, columns=COLUMNS)

    os.makedirs(STATS_DIR, exist_ok=True)
    for dataset_name, grp in new_df.groupby('dataset'):
        stats_file = os.path.join(STATS_DIR, f"edge_to_coverage_stats_{dataset_name}.csv")
        grp.to_csv(stats_file, index=False)
        print(f"\nSaved {len(grp)} rows → {stats_file}")


if __name__ == '__main__':
    main()
