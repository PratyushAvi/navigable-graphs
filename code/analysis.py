import glob
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

def main():

    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    adjLists = glob.glob(f"{SAVEPATH}/adj*")

    stats = []

    for file in tqdm(adjLists):
    
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

        stats.append([
            splits[3],
            splits[4],
            counter,
            np.mean(outDeg),
            np.mean(inDeg),
            np.median(outDeg),
            np.median(inDeg),
            np.min(outDeg),
            np.max(outDeg)
        ])

    pd.DataFrame(stats, columns=['dataset', 'metric', 'num points', 'mean out degree', 'mean in degree', 'median out degree', 'median in degree', 'min out degree', 'max out degree']).to_csv("/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/stats.csv")

if __name__ == '__main__':
    main()
