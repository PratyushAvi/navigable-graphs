import collections
import os
import numpy as np
import networkx as nx
from heapq import heappush, heappop
import argparse
import h5py
from scipy.spatial.distance import cdist
import pandas as pd
from tqdm import tqdm

def classicBeamSearch(source, target, G, d_q, b, k):
    """
    G:   nx.DiGraph
    d_q: (n,) array of squared euclidean distances from every point to target
    b:   beam width
    k:   number of nearest neighbours to return
    """
    D = set([source])
    C = [(d_q[source], source)] # min-heap
    B = [(-d_q[source], source)] # max-heap
    nodes_expanded = 0

    while C:
        dist, node = heappop(C)
        nodes_expanded += 1
        if len(B) == b and -1 * B[0][0] < dist:
            break

        for y in G.successors(node):
            if y not in D:
                D.add(y)
                if len(B) < b or -1 * B[0][0] > d_q[y]:
                    heappush(B, (-d_q[y], y))
                    heappush(C, (d_q[y], y))

                    if len(B) == b + 1:
                        heappop(B)

    for _ in range(b - k):
        heappop(B)

    return B, nodes_expanded, len(D)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj_list", required=True, help="Adjacency list file")
    parser.add_argument("--dataset", required=True, help="Dataset file")
    parser.add_argument("--save_path", required=True, help="Place to save CSV")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width")
    parser.add_argument("--step_size", type=float, default=1, help="Coverage decrement step size")
    parser.add_argument("--min_coverage", type=float, default=90, help="Minimum coverage amount")
    parser.add_argument("--tests", type=int, default=100, help="Number of tests")
    args = parser.parse_args()

    splits = args.dataset.split("/")[-1].split("-")

    DATASET = {
        'name': splits[0],
        'filepath': args.dataset,
        'adj_list': args.adj_list
    }

    print(f"Loading {DATASET}")
    data = h5py.File(DATASET['filepath'], 'r')

    X = data['train'][:]
    Y = data['test'][:]
    n = X.shape[0]

    print(f"Building networkx graphs...")

    coverage = [c / 100 for c in np.arange(100, args.min_coverage, -1 * args.step_size)]

    G = load_graphs(args.adj_list, n, coverage)

    print(f"Avg out-degrees\n--------------")
    for i, g in enumerate(G):
        print(f"{coverage[i] * 100:0.1f}% navigable: {g.number_of_edges() / n:0.2f}")
    print("--------------")

    RECALL_KS  = [1, 10, 100]
    K_search   = min(args.beam_width, 100)   # return up to 100 candidates
    beam_width = args.beam_width
    random_source = np.random.randint(0, X.shape[0])

    def run_search(query_vectors, query_indices=None):
        """
        query_vectors: (m, d) array of query points
        query_indices: if not None, indices into X (for train queries)
        Returns a dict: k -> DataFrame with per-query stats for that recall level.
        """
        # stats[k] holds per-query rows for recall@k
        stats = {k: collections.defaultdict(list) for k in RECALL_KS}

        for i, qvec in enumerate(tqdm(query_vectors)):
            d_q  = cdist(qvec[np.newaxis], X, metric='sqeuclidean').ravel()
            # ground truth top-100 (exclude self for train queries)
            sorted_idx = np.argsort(d_q)
            q_id = query_indices[i] if query_indices is not None else i
            tgt  = q_id if query_indices is not None else -1
            true_top = sorted_idx[:100]

            for gi, g in enumerate(G):
                result, expanded, seen = classicBeamSearch(
                    random_source, tgt, g, d_q, beam_width, K_search
                )
                # B is a max-heap of (-dist, node); sort ascending by dist
                # print(sorted(result, key=lambda x: -x[0]), true_top[0], tgt)
                returned = np.array([node for _, node in sorted(result, key=lambda x: -x[0])])

                for k in RECALL_KS:
                    recall = len(np.intersect1d(returned[:k], true_top[:k])) / k
                    stats[k][f'recall_{coverage[gi]}'].append(recall)
                    stats[k][f'seen_{coverage[gi]}'].append(seen)
                    stats[k][f'expanded_{coverage[gi]}'].append(expanded)

            for k in RECALL_KS:
                stats[k]['q'].append(q_id)
                stats[k]['beam_width'].append(beam_width)
                stats[k]['k'].append(k)

        return {k: pd.DataFrame(stats[k]) for k in RECALL_KS}

    def print_summary(dfs, label):
        for k in RECALL_KS:
            df = dfs[k]
            summary_dict = collections.defaultdict(list)
            summary_dict['metric'] = ['avg recall', 'avg nodes seen', 'avg nodes expanded']
            for gi in range(len(G)):
                summary_dict[f'G_{coverage[gi]}'] = [
                    df[f'recall_{coverage[gi]}'].mean(),
                    df[f'seen_{coverage[gi]}'].mean(),
                    df[f'expanded_{coverage[gi]}'].mean(),
                ]
            summary = pd.DataFrame(summary_dict)
            print(f"\n=== {label}  recall@{k} ===")
            print(summary.to_string(index=False))

    # --- Train queries (sampled from X, ground truth computed on the fly) ---
    train_indices = np.sort(np.random.choice(np.arange(X.shape[0]), size=args.tests, replace=False))
    print(f"\nSearching train queries (n={args.tests})...")
    dfs_train = run_search(X[train_indices], query_indices=train_indices)
    print_summary(dfs_train, "Train queries")

    # --- Summary CSV: one row per (dataset, beam_width, k), append if exists ---
    summary_rows = []
    for k in RECALL_KS:
        df = dfs_train[k]
        row = {
            'dataset':    DATASET['name'],
            'beam_width': beam_width,
            'k':          k,
        }
        for gi, g in enumerate(G):
            row[f'avg_edges_{coverage[gi]}']      = g.number_of_edges() / n
            row[f'train_recall_{coverage[gi]}']   = df[f'recall_{coverage[gi]}'].mean()
            row[f'train_seen_{coverage[gi]}']     = df[f'seen_{coverage[gi]}'].mean()
            row[f'train_expanded_{coverage[gi]}'] = df[f'expanded_{coverage[gi]}'].mean()
        summary_rows.append(row)

    summary_df  = pd.DataFrame(summary_rows)
    summary_path = f"{args.save_path}/beam_search_summary.csv"

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        mask = (existing['dataset'] == DATASET['name']) & (existing['beam_width'] == beam_width)
        if 'k' in existing.columns:
            mask = mask & existing['k'].isin(RECALL_KS)
        existing   = existing[~mask]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")

def load_graphs(adj_list_path, n, coverages):
    import ast

    G = [nx.DiGraph() for _ in range(len(coverages))]
    for g in G:
        g.add_nodes_from(range(n))

    with open(adj_list_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            space = line.index(' ')
            source       = int(line[:space])
            neighborhood = ast.literal_eval(line[space + 1:])   # [(neighbor, uncov), ...]

            for neighbor, uncov in neighborhood:
                for i, g in enumerate(G):
                    if uncov > (n * (1 - coverages[i])):
                        g.add_edge(source, neighbor)

    return G

if __name__ == '__main__':
    main()