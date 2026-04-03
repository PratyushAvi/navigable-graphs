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
    C = [(d_q[source], source)]
    B = [(d_q[source], source)]
    nodes_expanded = 0

    while C:
        dist, node = heappop(C)
        nodes_expanded += 1
        if len(B) == b and B[0][0] <= dist:
            break

        for y in G.successors(node):
            if y not in D:
                D.add(y)
                if len(B) < b or B[0][0] > d_q[y]:
                    heappush(B, (d_q[y], y))
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
    n = X.shape[0]

    print(f"Building networkx graphs...")
    G, G_99 = load_graphs(args.adj_list, n)

    queries = np.sort(np.random.choice(np.arange(X.shape[0]), size=10, replace=False))
    K = 10
    beam_width = 10
    random_source = np.random.randint(0, X.shape[0])

    results = {
        'q': [],
        'source': [],
        'beam_width': [],
        'number_of_results': [],
        'top_K_full': [],
        'top_K_partial': [],
        'relevant_full': [],
        'relevant_partial': [],
        'seen_full': [],
        'seen_partial': [],
        'expanded_full': [],
        'expanded_partial': []
    }

    print("Performing Search...")
    for q in tqdm(queries):
        # Distances from every point to query q — one vectorized BLAS call
        d_q = cdist(X[q:q+1], X, metric='sqeuclidean').ravel()

        # Top-100 ground truth for free from the same array
        top_100_neighbors = np.argsort(d_q)[:101]
        top_100_neighbors = top_100_neighbors[top_100_neighbors != q][:100]

        result_G,    expanded_G,    seen_G    = classicBeamSearch(random_source, q, G,    d_q, beam_width, K)
        result_G_99, expanded_G_99, seen_G_99 = classicBeamSearch(random_source, q, G_99, d_q, beam_width, K)

        nodes_G    = np.array([node for _, node in result_G])
        nodes_G_99 = np.array([node for _, node in result_G_99])

        rel_G    = np.intersect1d(nodes_G,    top_100_neighbors)
        rel_G_99 = np.intersect1d(nodes_G_99, top_100_neighbors)

        results['q'].append(q)
        results['source'].append(random_source)
        results['beam_width'].append(beam_width)
        results['number_of_results'].append(K)
        results['top_K_full'].append(nodes_G.tolist())
        results['top_K_partial'].append(nodes_G_99.tolist())
        results['relevant_full'].append(rel_G.tolist())
        results['relevant_partial'].append(rel_G_99.tolist())
        results['seen_full'].append(seen_G)
        results['seen_partial'].append(seen_G_99)
        results['expanded_full'].append(expanded_G)
        results['expanded_partial'].append(expanded_G_99)
    
    df = pd.DataFrame(results)
    print(df)
    df.to_csv(f"{args.adj_list}/beam_search_{DATASET['name']}.csv")

def load_graphs(adj_list_path, n):
    import ast

    G    = nx.DiGraph()
    G_99 = nx.DiGraph()
    G.add_nodes_from(range(n))
    G_99.add_nodes_from(range(n))

    threshold = 0.01 * n

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
                if uncov > threshold:
                    G_99.add_edge(source, neighbor)

    return G, G_99

if __name__ == '__main__':
    main()