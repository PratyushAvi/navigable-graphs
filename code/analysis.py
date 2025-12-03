import glob
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

def main():
    DATASETS = dict()
    dataset_records = pd.read_csv("/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d

    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    adjLists = glob.glob(f"{SAVEPATH}/adj*")

    stats = []

    for file in adjLists:
    
        G = nx.DiGraph()
    
        with open(file, 'r') as f:
            counter = 0
            for line in f:
                counter += 1
                points = [p for p in line.split(',')]
                for i in range(1, len(points)):
                    G.add_edge(points[0], points[i])
    
        splits = file.replace(".txt", "").split("-")
        outDeg = [d for _,d in G.out_degree() if d != 0]
        inDeg = [d for _,d in G.in_degree() if d != 0]
        print(sum(outDeg), sum(inDeg))

        stats.append([
            splits[3],
            splits[4],
            DATASETS[splits[3]]['dimensions'],
            counter,
            DATASETS[splits[3]]['train'],
            np.round(sum(outDeg)/counter, 2),
            np.median(outDeg),
            np.median(inDeg),
            np.min(outDeg),
            np.max(outDeg),
            np.min(inDeg),
            np.max(inDeg)
        ])

    pd.DataFrame(stats, columns=['dataset', 'metric', 'dimensions', 'points computed', 'total points', 'mean out degree', 'median out degree', 'median in degree', 'min out degree', 'max out degree', 'min in degree', 'max in degree']).to_csv("/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/stats.csv")

if __name__ == '__main__':
    main()
