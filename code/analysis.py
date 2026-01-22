import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import argparse
import gc

def main():
    parser = argparse.ArgumentParser(description='Process adjacency lists and compute graph statistics')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to process (e.g., spacev1b). If not specified, processes all incomplete datasets.')
    parser.add_argument('--chunk-size', type=int, default=10**6,
                        help='Number of lines to process at once (default: 1000000)')
    args = parser.parse_args()
    
    DATASETS = dict()
    dataset_records = pd.read_csv("/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    adjLists = glob.glob(f"{SAVEPATH}/adj*")
    
    # Load existing stats if available
    stats_file = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/stats.csv"
    if os.path.exists(stats_file):
        existing_stats = pd.read_csv(stats_file)
        stats_dict = {}
        for _, row in existing_stats.iterrows():
            key = (row['dataset'], row['metric'])
            stats_dict[key] = (row['points computed'], row['total points'])
    else:
        stats_dict = {}
    
    # Filter files that need processing
    files_to_process = []
    for file in adjLists:
        splits = file.replace(".txt", "").split("-")
        dataset_name = splits[3]
        metric = splits[4]
        
        if args.dataset is not None and dataset_name != args.dataset:
            continue
        
        key = (dataset_name, metric)
        
        if key not in stats_dict:
            files_to_process.append(file)
        else:
            computed, total = stats_dict[key]
            if computed < total:
                files_to_process.append(file)
    
    if args.dataset:
        print(f"Processing dataset: {args.dataset}")
    print(f"Found {len(files_to_process)} files to process out of {len(adjLists)} total files")
    
    if len(files_to_process) == 0:
        print("No files to process. All selected datasets are complete.")
        return
    
    stats = []
    
    for file in tqdm(files_to_process, desc="Processing adjacency lists"):
        splits = file.replace(".txt", "").split("-")
        dataset_name = splits[3]
        metric = splits[4]
        
        n_nodes = DATASETS[dataset_name]['train']
        print(f"\nProcessing {dataset_name}-{metric} with {n_nodes} nodes")
        
        # Use memory-mapped files for very large datasets
        degrees_path = '/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/degrees'
        os.makedirs(degrees_path, exist_ok=True)
        
        outDeg_file = f'{degrees_path}/{dataset_name}-{metric}-out-degrees.npy'
        inDeg_file = f'{degrees_path}/{dataset_name}-{metric}-in-degrees.npy'
        
        # Create memory-mapped arrays
        outDeg = np.memmap(outDeg_file, dtype='uint32', mode='w+', shape=(n_nodes,))
        inDeg = np.memmap(inDeg_file, dtype='uint32', mode='w+', shape=(n_nodes,))
        
        counter = 0
        chunk = []
        
        print(f"Reading adjacency list...")
        with open(file, 'r') as f:
            for line in tqdm(f, desc=f"Processing edges", leave=False):
                counter += 1
                points = [int(p.strip()) for p in line.strip().split(',')]
                chunk.append(points)
                
                # Process in chunks to avoid memory issues
                if len(chunk) >= args.chunk_size:
                    process_chunk(chunk, outDeg, inDeg, n_nodes)
                    chunk = []
                    if counter % (args.chunk_size * 10) == 0:
                        outDeg.flush()
                        inDeg.flush()
                        gc.collect()
            
            # Process remaining chunk
            if chunk:
                process_chunk(chunk, outDeg, inDeg, n_nodes)
                chunk = []
        
        print(f"Computed degrees for {counter} points")
        
        # Flush to disk
        outDeg.flush()
        inDeg.flush()
        
        print("Computing statistics...")
        # Compute statistics directly from memmap
        outDegNNZ = outDeg[outDeg != 0]
        
        if len(outDegNNZ) == 0:
            print(f"⚠ Warning: {dataset_name}-{metric} has no edges!")
            del outDeg, inDeg
            os.remove(outDeg_file)
            os.remove(inDeg_file)
            continue
        
        stats.append([
            dataset_name,
            metric,
            DATASETS[dataset_name]['dimensions'],
            counter,
            n_nodes,
            np.round(np.mean(outDegNNZ), 2),
            float(np.median(outDegNNZ)),
            float(np.median(inDeg[:])),
            int(np.min(outDegNNZ)),
            int(np.max(outDegNNZ)),
            int(np.min(inDeg[:])),
            int(np.max(inDeg[:]))
        ])
        
        print(f"✓ Completed {dataset_name}-{metric}: {counter} points, avg out-degree: {np.round(np.mean(outDegNNZ), 2)}")
        print(f"  Saved degrees to: {outDeg_file} and {inDeg_file}")
        
        # Clean up
        del outDeg, inDeg
        gc.collect()
    
    # Merge new stats with existing stats
    new_stats_df = pd.DataFrame(stats, columns=['dataset', 'metric', 'dimensions', 'points computed', 'total points', 
                                                  'mean out degree', 'median out degree', 'median in degree', 
                                                  'min out degree', 'max out degree', 'min in degree', 'max in degree'])
    
    if os.path.exists(stats_file):
        processed_keys = set((row[0], row[1]) for row in stats)
        existing_stats = existing_stats[~existing_stats.apply(lambda row: (row['dataset'], row['metric']) in processed_keys, axis=1)]
        combined_stats = pd.concat([existing_stats, new_stats_df], ignore_index=True)
    else:
        combined_stats = new_stats_df
    
    combined_stats.to_csv(stats_file, index=False)
    print(f"\nUpdated stats.csv with {len(new_stats_df)} entries")

def process_chunk(chunk, outDeg, inDeg, n_nodes):
    """Process a chunk of edges and update degree arrays"""
    for points in chunk:
        source = points[0]
        neighbors = points[1:]
        
        if source < n_nodes:
            outDeg[source] = len(neighbors)
        
        for neighbor in neighbors:
            if neighbor < n_nodes:
                inDeg[neighbor] += 1

if __name__ == '__main__':
    main()
