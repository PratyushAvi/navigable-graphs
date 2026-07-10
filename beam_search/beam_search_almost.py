"""Beam search over an almost-navigable graph.

Companion to almost_navigable_graphs/buildgraphs.py. That builder writes a plain
adjacency list -- "<source_id> [e_1, e_2, ...]" per line -- with no coverage /
uncov ranking (every edge is structural). This script loads that format into a
single CSR graph and runs the same beam search as beam_search.py, reporting
recall / nodes-seen / nodes-expanded over test queries.

The core classicBeamSearch is identical to beam_search.py; only graph loading
differs (one graph here vs. beam_search.py's per-coverage family).
"""
import argparse
import ast
import collections
import os
from heapq import heappush, heappop

import numpy as np
import h5py
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm


def classicBeamSearch(source, target, G, d_q, b, k):
    """
    G:   (indptr, neighbors) CSR adjacency. successors of u are
         neighbors[indptr[u]:indptr[u+1]].
    d_q: (n,) array of squared euclidean distances from every point to target
    b:   beam width
    k:   number of nearest neighbours to return
    """
    indptr, neighbors = G
    D = set([source])
    C = [(d_q[source], source)]       # min-heap
    B = [(-d_q[source], source)]      # max-heap
    nodes_expanded = 0

    while C:
        dist, node = heappop(C)
        nodes_expanded += 1
        if len(B) == b and -1 * B[0][0] < dist:
            break

        for y in neighbors[indptr[node]:indptr[node + 1]]:
            y = int(y)
            if y not in D:
                D.add(y)
                if len(B) < b or -1 * B[0][0] > d_q[y]:
                    heappush(B, (-d_q[y], y))
                    heappush(C, (d_q[y], y))
                    if len(B) == b + 1:
                        heappop(B)

    # Trim the beam down to the k closest (max-heap on -distance: pop farthest).
    # Never pop below what we have or below k, to avoid IndexError on sparse graphs.
    while len(B) > k:
        heappop(B)

    return B, nodes_expanded, len(D)


def load_graph(adj_list_path, n):
    """Load the plain adjacency list into a single CSR graph (indptr, neighbors).

    Each line is "<source_id> [e_1, e_2, ...]". Missing sources (no line) get an
    empty neighbour list.
    """
    adj = [[] for _ in range(n)]
    with open(adj_list_path, 'r') as f:
        for line in tqdm(f, desc="Loading graph"):
            line = line.strip()
            if not line:
                continue
            space = line.index(' ')
            source = int(line[:space])
            adj[source] = ast.literal_eval(line[space + 1:])   # [e_1, e_2, ...]

    degrees = np.fromiter((len(adj[u]) for u in range(n)), dtype=np.int64, count=n)
    indptr = np.empty(n + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    total = int(indptr[-1])
    neighbors = np.fromiter(
        (v for u in range(n) for v in adj[u]),
        dtype=np.int32, count=total,
    )
    return indptr, neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj_list", required=True,
                        help="Adjacency list file ('<src> [e_1, e_2, ...]' per line)")
    parser.add_argument("--dataset", required=True, help="Dataset hdf5 file (train/test)")
    parser.add_argument("--save_path", required=True, help="Directory to save the summary CSV into")
    parser.add_argument("--out_csv", default="beam_search_almost.csv",
                        help="Summary CSV filename (joined to --save_path)")
    parser.add_argument("--beam_widths", type=int, nargs='+', default=[1],
                        help="One or more beam widths")
    parser.add_argument("--tests", type=int, default=10000, help="Number of test queries")
    args = parser.parse_args()

    name = args.dataset.split("/")[-1].split("-")[0]
    print(f"Loading {name} from {args.dataset}")
    data = h5py.File(args.dataset, 'r')
    X = data['train'][:]
    Y = data['test'][:]
    n = X.shape[0]

    G = load_graph(args.adj_list, n)
    print(f"Graph loaded: {len(G[1])} edges, avg out-degree {len(G[1]) / n:.2f}")

    RECALL_KS     = [1, 10, 100]
    beam_widths   = sorted(args.beam_widths)
    random_source = np.random.randint(0, n)

    def run_search(query_vectors, query_indices=None):
        stats = {(bw, k): collections.defaultdict(list) for bw in beam_widths for k in RECALL_KS}
        for i, qvec in enumerate(tqdm(query_vectors)):
            d_q      = cdist(qvec[np.newaxis], X, metric='sqeuclidean').ravel()
            sorted_idx = np.argsort(d_q)
            q_id     = query_indices[i] if query_indices is not None else i
            tgt      = q_id if query_indices is not None else -1
            true_top = sorted_idx[:100]

            for bw in beam_widths:
                k_ret = min(bw, 100)
                result, expanded, seen = classicBeamSearch(
                    random_source, tgt, G, d_q, bw, k_ret
                )
                returned = np.array([node for _, node in sorted(result, key=lambda x: -x[0])])
                for k in RECALL_KS:
                    stats[(bw, k)]['relevant'].append(
                        len(np.intersect1d(returned[:k], true_top[:k]))
                    )
                    stats[(bw, k)]['seen'].append(seen)
                    stats[(bw, k)]['expanded'].append(expanded)

            for bw in beam_widths:
                for k in RECALL_KS:
                    stats[(bw, k)]['q'].append(q_id)
                    stats[(bw, k)]['beam_width'].append(bw)
                    stats[(bw, k)]['k'].append(k)

        return {key: pd.DataFrame(s) for key, s in stats.items()}

    n_tests = min(args.tests, Y.shape[0])
    test_indices = np.sort(np.random.choice(np.arange(Y.shape[0]), size=n_tests, replace=False))
    print(f"\nSearching test queries (n={n_tests}, beam_widths={beam_widths})...")
    dfs_test = run_search(Y[test_indices], query_indices=test_indices)

    summary_rows = []
    for bw in beam_widths:
        for k in RECALL_KS:
            df = dfs_test[(bw, k)]
            print(f"=== beam_width={bw}  recall@{k}: "
                  f"recall={df['relevant'].mean() / k:.4f}  "
                  f"seen={df['seen'].mean():.1f}  expanded={df['expanded'].mean():.1f}")
            summary_rows.append({
                'dataset': name,
                'beam_width': bw,
                'k': k,
                'avg_edges': len(G[1]) / n,
                'recall': df['relevant'].mean() / k,
                'seen': df['seen'].mean(),
                'expanded': df['expanded'].mean(),
            })

    KEY = ['dataset', 'beam_width', 'k']
    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(args.save_path, exist_ok=True)
    summary_path = os.path.join(args.save_path, args.out_csv)
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        summary_df = (pd.concat([existing, summary_df], ignore_index=True)
                        .drop_duplicates(subset=KEY, keep='last').reset_index(drop=True))
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path} ({len(summary_df)} rows)")


if __name__ == '__main__':
    main()