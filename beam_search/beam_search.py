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
        if len(B) == b and -1 * B[0][0] <= dist:
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
    Y_top_100 = data['neighbors'][:]
    n = X.shape[0]

    print(f"Building networkx graphs...")

    coverage = np.arange(0.8, 1, 0.25)

    G = load_graphs(args.adj_list, n, coverage)
    print(coverage, len(G), G)

    print(f"Avg out-degrees\n--------------")
    for i, g in enumerate(G):
        print(f"{coverage[i] * 100:>4}% navigable:{g.number_of_edges() / n:>3.2f}")
    print("--------------")

    K = 1
    beam_width = 1
    random_source = np.random.randint(0, X.shape[0])

    def run_search(query_vectors, query_indices=None):
        """
        query_vectors: (m, d) array of query points
        ground_truth:  (m, 100) array of true neighbor indices, or None (compute from X)
        query_indices: if not None, indices into X (for train queries); distances exclude self
        """
        stats = collections.defaultdict(lambda: [])

        for i, qvec in enumerate(tqdm(query_vectors)):
            d_q = cdist(qvec[np.newaxis], X, metric='sqeuclidean').ravel()

            top_K_neighbors = np.argsort(d_q)[:K]

            q_id = query_indices[i] if query_indices is not None else i
            tgt  = q_id if query_indices is not None else -1

            for i, g in enumerate(G):
                result, expanded, seen = classicBeamSearch(random_source, tgt, g, d_q, beam_width, K)
                nodes = np.array([node for _, node in result])
                
                relevant_nodes = np.intersect1d(nodes, top_K_neighbors)
                recall = len(G) / K
                
                stats['q'].append(q_id)
                stats['source'].append(random_source)
                stats['beam_width'].append(beam_width)
                stats['number_of_results'].append(K)
                stats[f'top_K_{coverage[i]}'].append(nodes.tolist())
                stats[f'relevant_{coverage[i]}'].append(relevant_nodes.tolist())
                stats[f'recall_{coverage[i]}'].append(recall)
                stats[f'seen_{coverage[i]}'].append(seen)
                stats[f'expanded_{coverage[i]}'].append(expanded)

        return pd.DataFrame(stats)

    def print_summary(df, label):
        summary_dict = collections.defaultdict(lambda: [])
        summary_dict['metric'] = ['avg recall', 'avg nodes seen', 'avg nodes expanded']
        for i, _ in enumerate(G):
            summary_dict[f'G_{coverage[i]}'] = [df[f'recall_{coverage[i]}'].mean(),
                     df[f'seen_{coverage[i]}'].mean(),       df[f'expanded_{coverage[i]}'].mean()],

        summary = pd.DataFrame(summary_dict)
        print(f"\n=== {label} ===")
        print(summary.to_string(index=False))

    # --- Train queries (sampled from X, ground truth computed on the fly) ---
    train_indices = np.sort(np.random.choice(np.arange(X.shape[0]), size=1000, replace=False))
    print(f"\nSearching train queries (n=1000)...")
    df_train = run_search(X[train_indices], query_indices=train_indices)
    print_summary(df_train, "Train queries")

    # --- Summary CSV (one row per run, append if exists) ---
    summary_row = {
        'dataset':     DATASET['name'],
        'beam_width':  beam_width,
        'num_results': K,
    }
    for i, g in enumerate(G):
        summary_row[f'avg_edges_{coverage[i]}'] = g.number_of_edges() / n
    for i, _ in enumerate(G):
        summary_row[f'train_recall_{coverage[i]}']   = df_train[f'recall_{coverage[i]}'].mean()
        summary_row[f'train_seen_{coverage[i]}']     = df_train[f'seen_{coverage[i]}'].mean()
        summary_row[f'train_expanded_{coverage[i]}'] = df_train[f'expanded_{coverage[i]}'].mean()

    summary_df = pd.DataFrame([summary_row])
    summary_path = f"{args.save_path}/beam_search_summary.csv"

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[existing['dataset'] != DATASET['name']]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)
    else:
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
                    if uncov > (n * coverages[i]):
                        g.add_edge(source, neighbor)

    return G

if __name__ == '__main__':
    main()