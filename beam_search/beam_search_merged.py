"""Beam search over a merged graph produced by graph_merge.py.

Same search as beam_search.py — classicBeamSearch is imported from there rather
than reimplemented, so results are directly comparable. The differences are in
the input and the output:

  * Input is a MERGE_BUILD adjacency ("<source> [n1, n2, ...]"), a plain neighbor
    list with no uncov counts. There is no coverage sweep to run: the merged
    graph is a single fixed graph, so `coverage` is not part of the output key.
  * Output adds graph degree statistics (mean/max/min) alongside the search
    statistics, written to beam_search_<split>_merged.csv.

Run with --split train, --split test, or --split both.
"""

import argparse
import ast
import collections
import os

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

from beam_search import classicBeamSearch

RECALL_KS = [1, 10, 100]

# One row per (dataset, split, beam_width, k); each run upserts on that key.
KEY_COLS = ['dataset', 'split', 'beam_width', 'k']
VAL_COLS = [
    'queries',
    'mean_recall', 'mean_seen', 'mean_expanded',
    'n_points', 'n_edges',
    'mean_degree', 'max_degree', 'min_degree',
    'isolated_points',
]


def load_merged_graph(adj_list_path, n):
    """Load a MERGE_BUILD adjacency into CSR form: (indptr, neighbors).

    Successors of u are neighbors[indptr[u]:indptr[u+1]] — the same layout
    beam_search.load_graphs produces, so classicBeamSearch works unchanged.

    Accepts plain lists ("[n1, n2, ...]"). Tuple-form lines from simulrun.py are
    rejected rather than silently misread: those need the coverage-threshold
    logic in beam_search.load_graphs, which this script deliberately omits.
    """
    adj = [[] for _ in range(n)]
    seen_sources = set()

    with open(adj_list_path, 'r') as f:
        for lineno, line in enumerate(tqdm(f, desc="Reading adjacency"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                space = line.index(' ')
                source = int(line[:space])
                neighborhood = ast.literal_eval(line[space + 1:])
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"{os.path.basename(adj_list_path)}:{lineno}: cannot parse "
                    f"line (expected '<source> [n1, n2, ...]'): {e}")

            if neighborhood and isinstance(neighborhood[0], tuple):
                raise ValueError(
                    f"{os.path.basename(adj_list_path)}:{lineno}: this looks like "
                    f"a simulrun.py adjacency with (neighbor, uncov) tuples. "
                    f"beam_search_merged.py expects a merged graph from "
                    f"graph_merge.py; use beam_search.py for tuple-form files.")

            if source in seen_sources:
                raise ValueError(
                    f"{os.path.basename(adj_list_path)}:{lineno}: duplicate "
                    f"source {source}")
            if not 0 <= source < n:
                raise ValueError(
                    f"{os.path.basename(adj_list_path)}:{lineno}: source {source} "
                    f"is outside the dataset ({n} points)")

            seen_sources.add(source)
            adj[source] = [int(v) for v in neighborhood]

    degrees = np.fromiter((len(a) for a in adj), dtype=np.int64, count=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    total = int(indptr[-1])
    neighbors = np.fromiter(
        (v for a in adj for v in a), dtype=np.int32, count=total,
    )
    return (indptr, neighbors), degrees


def graph_stats(degrees):
    """Degree statistics for the merged graph.

    mean is over all n points (so it matches n_edges / n_points), while max/min
    are plain extremes over the out-degree distribution. A min of 0 means some
    point has no outgoing edges — reported separately as isolated_points since
    those are unreachable-from and will drag recall down.
    """
    return {
        'n_points':        int(len(degrees)),
        'n_edges':         int(degrees.sum()),
        'mean_degree':     float(degrees.mean()),
        'max_degree':      int(degrees.max()),
        'min_degree':      int(degrees.min()),
        'isolated_points': int((degrees == 0).sum()),
    }


def run_search(query_vectors, X, G, beam_widths, source, query_indices=None):
    """Beam search for every query at every beam width.

    Returns {(beam_width, k): DataFrame of per-query rows}. Ground truth is the
    exact top-100 by squared euclidean distance, recomputed per query.
    """
    stats = {(bw, k): collections.defaultdict(list)
             for bw in beam_widths for k in RECALL_KS}

    for i, qvec in enumerate(tqdm(query_vectors, desc="Searching")):
        d_q = cdist(qvec[np.newaxis], X, metric='sqeuclidean').ravel()
        true_top = np.argsort(d_q)[:100]
        q_id = query_indices[i] if query_indices is not None else i
        tgt = q_id if query_indices is not None else -1

        for bw in beam_widths:
            k_ret = min(bw, 100)
            result, expanded, seen = classicBeamSearch(source, tgt, G, d_q, bw, k_ret)
            returned = np.array([node for _, node in sorted(result, key=lambda x: -x[0])])

            for k in RECALL_KS:
                s = stats[(bw, k)]
                s['q'].append(q_id)
                s['beam_width'].append(bw)
                s['k'].append(k)
                s['relevant'].append(len(np.intersect1d(returned[:k], true_top[:k])))
                s['seen'].append(seen)
                s['expanded'].append(expanded)

    return {key: pd.DataFrame(s) for key, s in stats.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Beam search over a merged graph from graph_merge.py, "
                    "reporting search and graph-degree statistics.")
    parser.add_argument("--adj_list", required=True,
                        help="MERGE_BUILD adjacency file from graph_merge.py")
    parser.add_argument("--dataset", required=True, help="Dataset HDF5 file")
    parser.add_argument("--save_path", required=True,
                        help="Directory to write beam_search_<split>_merged.csv into")
    parser.add_argument("--beam_widths", type=int, nargs='+', default=[1],
                        help="One or more beam widths")
    parser.add_argument("--tests", type=int, default=10000,
                        help="Number of queries to sample per split")
    parser.add_argument("--split", choices=['test', 'train', 'both'], default='test',
                        help="Which queries to run; 'both' writes one CSV per split "
                             "(default: test)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for query sampling and start-node choice, for "
                             "reproducible runs")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    dataset_name = os.path.basename(args.dataset).split("-")[0]
    print(f"Loading {args.dataset}")
    data = h5py.File(args.dataset, 'r')
    X = data['train'][:]
    n = X.shape[0]

    print(f"Loading merged graph from {args.adj_list}")
    try:
        G, degrees = load_merged_graph(args.adj_list, n)
    except ValueError as e:
        parser.error(str(e))

    gstats = graph_stats(degrees)
    print(f"\nGraph: {gstats['n_points']:,} points, {gstats['n_edges']:,} edges")
    print(f"Degree — mean {gstats['mean_degree']:.2f}  "
          f"max {gstats['max_degree']}  min {gstats['min_degree']}")
    if gstats['isolated_points']:
        print(f"WARNING: {gstats['isolated_points']:,} points have no outgoing "
              f"edges and cannot be searched from")

    beam_widths = sorted(args.beam_widths)
    # One start node shared by every query and beam width, as in beam_search.py.
    random_source = int(np.random.randint(0, n))
    print(f"Start node: {random_source}")

    splits = ['test', 'train'] if args.split == 'both' else [args.split]

    for split in splits:
        if split == 'train':
            pool, query_source = X, X
        else:
            pool = data['test'][:]
            query_source = pool

        if args.tests > pool.shape[0]:
            parser.error(f"--tests {args.tests} exceeds the {split} split size "
                         f"({pool.shape[0]})")

        indices = np.sort(np.random.choice(np.arange(pool.shape[0]),
                                           size=args.tests, replace=False))
        # Train queries are points in the graph, so their id is the node id and
        # the search can terminate on it; test queries are external (tgt = -1).
        query_indices = indices if split == 'train' else None

        print(f"\nSearching {split} queries (n={args.tests}, "
              f"beam_widths={beam_widths})...")
        dfs = run_search(query_source[indices], X, G, beam_widths,
                         random_source, query_indices=query_indices)

        rows = []
        for bw in beam_widths:
            for k in RECALL_KS:
                df = dfs[(bw, k)]
                rows.append({
                    'dataset':      dataset_name,
                    'split':        split,
                    'beam_width':   bw,
                    'k':            k,
                    'queries':      len(df),
                    # recall@k is the fraction of the true top-k that was returned
                    'mean_recall':  df['relevant'].mean() / k,
                    'mean_seen':    df['seen'].mean(),
                    'mean_expanded': df['expanded'].mean(),
                    **gstats,
                })

        summary_df = pd.DataFrame(rows, columns=KEY_COLS + VAL_COLS)
        out_path = os.path.join(args.save_path,
                                f"beam_search_{split}_merged.csv")

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, summary_df], ignore_index=True)
            summary_df = combined.drop_duplicates(
                subset=KEY_COLS, keep='last').reset_index(drop=True)

        summary_df.to_csv(out_path, index=False)
        print(f"\n=== {split} queries ===")
        print(pd.DataFrame(rows)[
            ['beam_width', 'k', 'mean_recall', 'mean_seen', 'mean_expanded']
        ].to_string(index=False))
        print(f"Written to {out_path} ({len(summary_df)} rows)")


if __name__ == '__main__':
    main()
