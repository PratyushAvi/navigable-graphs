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
    G, G_99, G_90 = load_graphs(args.adj_list, n)

    avg_deg_G    = G.number_of_edges()    / n
    avg_deg_G_99 = G_99.number_of_edges() / n
    avg_deg_G_90 = G_90.number_of_edges() / n
    print(f"Avg out-degree — G: {avg_deg_G:.2f} | G_99: {avg_deg_G_99:.2f} | G_90: {avg_deg_G_90:.2f}")

    K = 100
    beam_width = 150
    random_source = np.random.randint(0, X.shape[0])

    def run_search(query_vectors, ground_truth, query_indices=None):
        """
        query_vectors: (m, d) array of query points
        ground_truth:  (m, 100) array of true neighbor indices, or None (compute from X)
        query_indices: if not None, indices into X (for train queries); distances exclude self
        """
        results = {
            'q': [],
            'source': [],
            'beam_width': [],
            'number_of_results': [],
            'top_K_full': [],
            'top_K_99': [],
            'top_K_90': [],
            'relevant_full': [],
            'relevant_99': [],
            'relevant_90': [],
            'precision_full': [],
            'precision_99': [],
            'precision_90': [],
            'recall_full': [],
            'recall_99': [],
            'recall_90': [],
            'seen_full': [],
            'seen_99': [],
            'seen_90': [],
            'expanded_full': [],
            'expanded_99': [],
            'expanded_90': []
        }

        for i, qvec in enumerate(tqdm(query_vectors)):
            d_q = cdist(qvec[np.newaxis], X, metric='sqeuclidean').ravel()

            if ground_truth is not None:
                top_100_neighbors = ground_truth[i][:100]
            else:
                q = query_indices[i]
                top_100_neighbors = np.argsort(d_q)[:101]
                top_100_neighbors = top_100_neighbors[top_100_neighbors != q][:100]

            q_id = query_indices[i] if query_indices is not None else i
            tgt  = q_id if query_indices is not None else -1

            result_G,    expanded_G,    seen_G    = classicBeamSearch(random_source, tgt, G,    d_q, beam_width, K)
            result_G_99, expanded_G_99, seen_G_99 = classicBeamSearch(random_source, tgt, G_99, d_q, beam_width, K)
            result_G_90, expanded_G_90, seen_G_90 = classicBeamSearch(random_source, tgt, G_90, d_q, beam_width, K)

            nodes_G    = np.array([node for _, node in result_G])
            nodes_G_99 = np.array([node for _, node in result_G_99])
            nodes_G_90 = np.array([node for _, node in result_G_90])

            rel_G    = np.intersect1d(nodes_G,    top_100_neighbors)
            rel_G_99 = np.intersect1d(nodes_G_99, top_100_neighbors)
            rel_G_90 = np.intersect1d(nodes_G_90, top_100_neighbors)

            prec_G    = len(rel_G)    / K
            prec_G_99 = len(rel_G_99) / K
            prec_G_90 = len(rel_G_90) / K
            rec_G     = len(rel_G)    / 100
            rec_G_99  = len(rel_G_99) / 100
            rec_G_90  = len(rel_G_90) / 100

            results['q'].append(q_id)
            results['source'].append(random_source)
            results['beam_width'].append(beam_width)
            results['number_of_results'].append(K)
            results['top_K_full'].append(nodes_G.tolist())
            results['top_K_99'].append(nodes_G_99.tolist())
            results['top_K_90'].append(nodes_G_90.tolist())
            results['relevant_full'].append(rel_G.tolist())
            results['relevant_99'].append(rel_G_99.tolist())
            results['relevant_90'].append(rel_G_90.tolist())
            results['precision_full'].append(prec_G)
            results['precision_99'].append(prec_G_99)
            results['precision_90'].append(prec_G_90)
            results['recall_full'].append(rec_G)
            results['recall_99'].append(rec_G_99)
            results['recall_90'].append(rec_G_90)
            results['seen_full'].append(seen_G)
            results['seen_99'].append(seen_G_99)
            results['seen_90'].append(seen_G_90)
            results['expanded_full'].append(expanded_G)
            results['expanded_99'].append(expanded_G_99)
            results['expanded_90'].append(expanded_G_90)

        return pd.DataFrame(results)

    def print_summary(df, label):
        summary = pd.DataFrame({
            'metric': ['avg precision', 'avg recall', 'avg nodes seen', 'avg nodes expanded'],
            'G':    [df['precision_full'].mean(), df['recall_full'].mean(),
                     df['seen_full'].mean(),       df['expanded_full'].mean()],
            'G_99': [df['precision_99'].mean(),   df['recall_99'].mean(),
                     df['seen_99'].mean(),         df['expanded_99'].mean()],
            'G_90': [df['precision_90'].mean(),   df['recall_90'].mean(),
                     df['seen_90'].mean(),         df['expanded_90'].mean()],
        })
        print(f"\n=== {label} ===")
        print(summary.to_string(index=False))

    # --- Train queries (sampled from X, ground truth computed on the fly) ---
    train_indices = np.sort(np.random.choice(np.arange(X.shape[0]), size=1000, replace=False))
    print(f"\nSearching train queries (n=1000)...")
    df_train = run_search(X[train_indices], ground_truth=None, query_indices=train_indices)
    print_summary(df_train, "Train queries")
    df_train.to_csv(f"{args.save_path}/beam_search_{DATASET['name']}_train.csv", index=False)

    # --- Test queries (Y, ground truth from Y_top_100) ---
    print(f"\nSearching test queries (n={Y.shape[0]})...")
    df_test = run_search(Y, ground_truth=Y_top_100, query_indices=None)
    print_summary(df_test, "Test queries")
    df_test.to_csv(f"{args.save_path}/beam_search_{DATASET['name']}_test.csv", index=False)

    # --- Summary CSV (one row per run, append if exists) ---
    summary_row = {
        'dataset':           DATASET['name'],
        'beam_width':        beam_width,
        'num_results':       K,
        'avg_edges':         avg_deg_G,
        'avg_edges_99':      avg_deg_G_99,
        'avg_edges_90':      avg_deg_G_90,
        # train
        'train_recall_G':    df_train['recall_full'].mean(),
        'train_recall_99':   df_train['recall_99'].mean(),
        'train_recall_90':   df_train['recall_90'].mean(),
        'train_seen_G':      df_train['seen_full'].mean(),
        'train_seen_99':     df_train['seen_99'].mean(),
        'train_seen_90':     df_train['seen_90'].mean(),
        'train_expanded_G':  df_train['expanded_full'].mean(),
        'train_expanded_99': df_train['expanded_99'].mean(),
        'train_expanded_90': df_train['expanded_90'].mean(),
        # test
        'test_recall_G':     df_test['recall_full'].mean(),
        'test_recall_99':    df_test['recall_99'].mean(),
        'test_recall_90':    df_test['recall_90'].mean(),
        'test_seen_G':       df_test['seen_full'].mean(),
        'test_seen_99':      df_test['seen_99'].mean(),
        'test_seen_90':      df_test['seen_90'].mean(),
        'test_expanded_G':   df_test['expanded_full'].mean(),
        'test_expanded_99':  df_test['expanded_99'].mean(),
        'test_expanded_90':  df_test['expanded_90'].mean(),
    }
    summary_path = f"{args.save_path}/beam_search_summary.csv"
    summary_df = pd.DataFrame([summary_row])
    write_header = not os.path.exists(summary_path)
    summary_df.to_csv(summary_path, mode='a', header=write_header, index=False)
    print(f"\nSummary appended to {summary_path}")
    

def load_graphs(adj_list_path, n):
    import ast

    G    = nx.DiGraph()
    G_99 = nx.DiGraph()
    G_90 = nx.DiGraph()
    G.add_nodes_from(range(n))
    G_99.add_nodes_from(range(n))
    G_90.add_nodes_from(range(n))

    threshold_99 = 0.01 * n
    threshold_90 = 0.10 * n

    with open(adj_list_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            space = line.index(' ')
            source       = int(line[:space])
            neighborhood = ast.literal_eval(line[space + 1:])   # [(neighbor, uncov), ...]

            for neighbor, uncov in neighborhood:
                G.add_edge(source, neighbor)
                if uncov > threshold_99:
                    G_99.add_edge(source, neighbor)
                if uncov > threshold_90:
                    G_90.add_edge(source, neighbor)

    return G, G_99, G_90

if __name__ == '__main__':
    main()