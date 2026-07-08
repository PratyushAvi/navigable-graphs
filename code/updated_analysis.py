import ast
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
from collections import defaultdict


COLUMNS = ['dataset', 'metric', 'method', 'dimensions', 'points computed', 'total points', 'coverage',
           'mean out degree', 'median out degree', 'median in degree',
           'min out degree', 'max out degree', 'min in degree', 'max in degree']


def main():
    parser = argparse.ArgumentParser(description='Compute graph degree stats at multiple coverage levels')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to process. If omitted, processes all incomplete datasets. '
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
    parser.add_argument('--min-coverage', type=float, default=90.0,
                        help='Minimum coverage level in percent (default: 90.0)')
    parser.add_argument('--max-coverage', type=float, default=100.0,
                        help='Maximum coverage level in percent (default: 100.0)')
    parser.add_argument('--step-size', type=float, default=0.5,
                        help='Step size between coverage levels in percent (default: 0.5)')
    parser.add_argument('--total-points', type=int, required=True,
                        help='Size of the full dataset that coverage is measured against '
                             '(the universe uncov counts range over, e.g. 1000000000 for bigann)')
    parser.add_argument('--dimensions', type=int, default=-1,
                        help='Vector dimensionality, recorded in the output for reference (default: -1)')
    args = parser.parse_args()

    # Build sorted list of coverage levels
    coverages = []
    c = args.min_coverage
    while c <= args.max_coverage + 1e-9:
        coverages.append(round(c, 4))
        c = round(c + args.step_size, 10)
    n_cov = len(coverages)
    print(f"Coverage levels ({n_cov}): {coverages[0]}% to {coverages[-1]}% step {args.step_size}%")

    SAVEPATH = "/scratch/pa2439/ANN-Search/navigable_graph_results/new_results"
    stats_file = "/scratch/pa2439/ANN-Search/navigable_graph_results/coverage_stats.csv"

    # Load existing stats (used for the merge at the end, and for skipping
    # already-complete combos when scanning SAVEPATH).
    if os.path.exists(stats_file):
        existing_stats = pd.read_csv(stats_file)
        if 'method' not in existing_stats.columns:
            existing_stats['method'] = 'robust-prune'
    else:
        existing_stats = pd.DataFrame()

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

        # Determine which (dataset, metric, method) combos are already fully done
        done_keys = set()
        if not existing_stats.empty:
            cov_set_required = set(coverages)
            for key, grp in existing_stats.groupby(['dataset', 'metric', 'method']):
                if cov_set_required.issubset(set(grp['coverage'])):
                    done_keys.add(key)

        files_to_process = []
        for file, method in adjLists:
            dataset_name, metric = parse_filename(file, method)
            if args.dataset is not None and dataset_name != args.dataset:
                continue
            if (dataset_name, metric, method) not in done_keys:
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

        # uncov_left threshold for each coverage level (non-increasing)
        # coverage c% is achieved when uncov_left <= (1 - c/100) * n_nodes.
        # Nudge up by 1e-6 so a source sitting exactly on a boundary (integer
        # uncov == threshold) isn't pushed to the next edge by float rounding of
        # (1 - c/100) * n_nodes (e.g. 80% of 10 evaluates to 1.9999999999999996).
        thresholds = [(1.0 - c / 100.0) * n_nodes + 1e-6 for c in coverages]

        # out_deg_per_source[i] = list of n_cov ints: edges needed by source i at each coverage level
        out_deg_per_source = []
        # in_deg[neighbor] = uint32 array of shape (n_cov,): in-degree at each coverage level
        in_deg = defaultdict(lambda: np.zeros(n_cov, dtype=np.uint32))

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

                # Single pass: compute out_deg and update in_deg simultaneously.
                # Before the while loop each iteration, cov_ptr is the first coverage
                # level not yet satisfied, so the current edge is included at all
                # levels c_idx >= cov_ptr (they still need more edges).
                #
                # uncov on tuple edge_idx is the uncov *after* that edge takes
                # effect (see fix_adj_list_uncov.py / distributed_robust_prune.py).
                # So uncov <= threshold at edge_idx means edges 0..edge_idx, i.e.
                # edge_idx + 1 edges, achieve that coverage level.
                out_deg_c = [len(neighborhood)] * n_cov
                cov_ptr = 0
                for edge_idx, (neighbor, uncov) in enumerate(neighborhood):
                    first_c = cov_ptr  # coverage levels [first_c:] include this edge
                    while cov_ptr < n_cov and uncov <= thresholds[cov_ptr]:
                        out_deg_c[cov_ptr] = edge_idx + 1
                        cov_ptr += 1
                    in_deg[neighbor][first_c:] += 1
                    if cov_ptr == n_cov:
                        break

                out_deg_per_source.append(out_deg_c)

        if counter == 0:
            print(f"  No sources found — skipping")
            continue

        print(f"  Processed {counter} sources, {len(in_deg)} unique neighbors seen")

        # --- Aggregate stats per coverage level ---
        out_deg_array = np.array(out_deg_per_source, dtype=np.int32)  # (counter, n_cov)
        in_deg_array  = np.array(list(in_deg.values()), dtype=np.int32) if in_deg else np.empty((0, n_cov), dtype=np.int32)

        for c_idx, cov in enumerate(coverages):
            out_c = out_deg_array[:, c_idx]
            out_nnz = out_c[out_c > 0]

            in_c = in_deg_array[:, c_idx] if len(in_deg_array) else np.array([], dtype=np.int32)
            in_nnz = in_c[in_c > 0]

            all_new_rows.append([
                dataset_name,
                metric,
                method,
                args.dimensions,
                counter,
                n_nodes,
                cov,
                float(np.round(np.mean(out_nnz),   2)) if len(out_nnz) else 0.0,
                float(np.median(out_nnz))               if len(out_nnz) else 0.0,
                float(np.median(in_nnz))                if len(in_nnz)  else 0.0,
                int(np.min(out_nnz))                    if len(out_nnz) else 0,
                int(np.max(out_nnz))                    if len(out_nnz) else 0,
                int(np.min(in_nnz))                     if len(in_nnz)  else 0,
                int(np.max(in_nnz))                     if len(in_nnz)  else 0,
            ])

        print(f"  Computed stats for {n_cov} coverage levels")

    # --- Save to CSV ---
    new_df = pd.DataFrame(all_new_rows, columns=COLUMNS)

    if not existing_stats.empty and all_new_rows:
        # Append new coverages, keeping all existing ones. Only overwrite an
        # existing row when it has the same (dataset, metric, method, coverage)
        # as a freshly computed row — those duplicates take the new values.
        key_cols = ['dataset', 'metric', 'method', 'coverage']
        new_keys = set(map(tuple, new_df[key_cols].itertuples(index=False, name=None)))
        mask = existing_stats[key_cols].apply(
            lambda row: tuple(row) in new_keys, axis=1
        )
        combined = pd.concat([existing_stats[~mask], new_df], ignore_index=True)
    elif not existing_stats.empty:
        combined = existing_stats
    else:
        combined = new_df

    combined.to_csv(stats_file, index=False)
    print(f"\nSaved {len(new_df)} new rows → {stats_file} ({len(combined)} total rows)")


if __name__ == '__main__':
    main()
