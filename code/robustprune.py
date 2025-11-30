import numpy as np
import cupy as cp
import h5py
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import rankdata
import psutil
from utils import *
import pandas as pd
import os


def main():
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    DATASETS = pd.read_csv("/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    DATASETS = sorted(DATASETS, key=lambda x: x['train']) 

    done = set([s.split('-')[2] for s in os.listdir(SAVEPATH)])

    for DATASET in DATASETS:
        if DATASET['name'] in done:
            print('skipping', DATASET['name'])
            continue
        #if DATASET['metric'] != 'euclidean':
        #   print("Skipping", DATASET['name'])
        #  continue
         
        print(f"Building graph on {DATASET['name']}")
        data = h5py.File(DATASET['filepath'], 'r')['train']
        
        dataset = cp.asarray(data[:])
        
        if DATASET['train'] < 1000000:
            robustPruneGraph = memBuildRobustPruneGraph(dataset)
        
        else:
            continue
            points = np.random.choice(DATASET['train'], 60000, replace=False)
            robustPruneGraph = smallSampleBuildRobustPruneGraph(dataset, points)
        
        n = len(robustPruneGraph)
        saveGraph(n, robustPruneGraph, f"{SAVEPATH}/adj-list-{DATASET['name']}-{n}-vertices-euclidean")
    
if __name__ == '__main__':
    main()
