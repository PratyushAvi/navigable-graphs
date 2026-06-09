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

    # Trim the beam down to the k closest. B is a max-heap on -distance, so each
    # heappop removes the current FARTHEST. The search may have reached fewer than
    # b distinct nodes (sparse / low-coverage graph), so never pop more than we have
    # and never below k — otherwise heappop on an empty B raises IndexError.
    while len(B) > k:
        heappop(B)

    return B, nodes_expanded, len(D)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj_list", required=True, help="Adjacency list file")
    parser.add_argument("--dataset", required=True, help="Dataset file")
    parser.add_argument("--save_path", required=True, help="Place to save CSV")
    parser.add_argument("--beam_widths", type=int, nargs='+', default=[1], help="One or more beam widths")
    parser.add_argument("--step_size", type=float, default=1, help="Coverage decrement step size")
    parser.add_argument("--min_coverage", type=float, default=90, help="Minimum coverage amount")
    parser.add_argument("--tests", type=int, default=10000, help="Number of tests")
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

    RECALL_KS     = [1, 10, 100]
    beam_widths   = sorted(args.beam_widths)
    K_search      = min(max(beam_widths), 100)   # return up to 100 candidates; same pass for all widths
    random_source = np.random.randint(0, X.shape[0])

    def run_search(query_vectors, query_indices=None):
        """
        query_vectors: (m, d) array of query points
        query_indices: if not None, indices into X (for train queries)
        Returns a dict: (beam_width, k) -> DataFrame with per-query stats.
        """
        stats = {(bw, k): collections.defaultdict(list) for bw in beam_widths for k in RECALL_KS}

        for i, qvec in enumerate(tqdm(query_vectors)):
            d_q      = cdist(qvec[np.newaxis], X, metric='sqeuclidean').ravel()
            sorted_idx = np.argsort(d_q)
            q_id     = query_indices[i] if query_indices is not None else i
            tgt      = q_id if query_indices is not None else -1
            true_top = sorted_idx[:100]

            for gi, g in enumerate(G):
                for bw in beam_widths:
                    k_ret   = min(bw, 100)
                    result, expanded, seen = classicBeamSearch(
                        random_source, tgt, g, d_q, bw, k_ret
                    )
                    returned = np.array([node for _, node in sorted(result, key=lambda x: -x[0])])

                    for k in RECALL_KS:
                        stats[(bw, k)][f'relevant_{coverage[gi]}'].append(
                            len(np.intersect1d(returned[:k], true_top[:k]))
                        )
                        stats[(bw, k)][f'seen_{coverage[gi]}'].append(seen)
                        stats[(bw, k)][f'expanded_{coverage[gi]}'].append(expanded)

            for bw in beam_widths:
                for k in RECALL_KS:
                    stats[(bw, k)]['q'].append(q_id)
                    stats[(bw, k)]['beam_width'].append(bw)
                    stats[(bw, k)]['k'].append(k)

        return {key: pd.DataFrame(s) for key, s in stats.items()}

    def print_summary(dfs, label):
        for bw in beam_widths:
            for k in RECALL_KS:
                df = dfs[(bw, k)]
                summary_dict = collections.defaultdict(list)
                summary_dict['metric'] = ['avg recall', 'avg nodes seen', 'avg nodes expanded']
                for gi in range(len(G)):
                    summary_dict[f'G_{coverage[gi]}'] = [
                        df[f'relevant_{coverage[gi]}'].mean() / k,
                        df[f'seen_{coverage[gi]}'].mean(),
                        df[f'expanded_{coverage[gi]}'].mean(),
                    ]
                summary = pd.DataFrame(summary_dict)
                print(f"\n=== {label}  beam_width={bw}  recall@{k} ===")
                print(summary.to_string(index=False))

    # --- Train queries (sampled from X, ground truth computed on the fly) ---
    # train_indices = np.sort(np.random.choice(np.arange(X.shape[0]), size=args.tests, replace=False))
    # print(f"\nSearching train queries (n={args.tests}, beam_widths={beam_widths})...")
    # dfs_train = run_search(X[train_indices], query_indices=train_indices)
    # print_summary(dfs_train, "Train queries")

    # --- Test queries (sampled from Y, ground truth computed on the fly) ---
    test_indices = np.sort(np.random.choice(np.arange(Y.shape[0]), size=args.tests, replace=False))
    print(f"\nSearching test queries (n={args.tests}, beam_widths={beam_widths})...")
    dfs_test = run_search(Y[test_indices], query_indices=test_indices)
    print_summary(dfs_test, "Test queries")

    # --- Summary CSV: one row per (dataset, beam_width, k), append if exists ---
    summary_rows = []
    for bw in beam_widths:
        for k in RECALL_KS:
            df = dfs_test[(bw, k)]
            row = {
                'dataset':    DATASET['name'],
                'beam_width': bw,
                'k':          k,
            }
            for gi, g in enumerate(G):
                row[f'avg_edges_{coverage[gi]}']        = g.number_of_edges() / n
                row[f'train_relevant_{coverage[gi]}']   = df[f'relevant_{coverage[gi]}'].mean()
                row[f'train_seen_{coverage[gi]}']       = df[f'seen_{coverage[gi]}'].mean()
                row[f'train_expanded_{coverage[gi]}']   = df[f'expanded_{coverage[gi]}'].mean()
            summary_rows.append(row)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = f"{args.save_path}/beam_search_summary_test_sc.csv"

    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        mask = (existing['dataset'] == DATASET['name']) & (existing['beam_width'].isin(beam_widths))
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

            # `uncov` is the count of points STILL uncovered AFTER adding this edge.
            # Greedy order => uncov is non-increasing along the list. To reach
            # coverage c we must keep every edge UP TO AND INCLUDING the one that first
            # brings uncovered <= n*(1-c). Keying off the post-add `uncov` (the old
            # `uncov > threshold`) drops exactly that crossing edge, leaving some
            # sources with too few (or zero) edges. Instead key off the uncovered
            # count BEFORE each edge: keep the edge while the previous count exceeded
            # the threshold. For the first edge that count is n-1, so every node keeps
            # at least one edge for any coverage < 100%.
            prev_uncov = n - 1   # only the source is covered before any edge is added
            for neighbor, uncov in neighborhood:
                for i, g in enumerate(G):
                    if prev_uncov > (n * (1 - coverages[i])):
                        g.add_edge(source, neighbor)
                prev_uncov = uncov

    return G

if __name__ == '__main__':
    main()
