import ast
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
    dataset_records = pd.read_csv("/scratch/pa2439/ANN-Search/navigable_graph_results/datasets.csv").to_dict('records')
    for d in dataset_records:
        DATASETS[d['name']] = d
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/new_results"
    adjLists = glob.glob(f"{SAVEPATH}/adj*")
    
    # Load existing stats if available
    stats_file = "/scratch/pa2439/ANN-Search/navigable_graph_results/99p_stats.csv"
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
        edges_to_99pct_sum = 0   # accumulate across all nodes for this file

        # spacev1b adj list lines have no source prefix; sources come from computed.txt in order
        spacev1b_sources = None
        if dataset_name == 'spacev1b':
            computed_txt = f"{SAVEPATH}/spacev1b-euclidean-computed.txt"
            with open(computed_txt, 'r') as cf:
                spacev1b_sources = [int(p.strip()) for p in cf if p.strip()]
            print(f"spacev1b: {len(spacev1b_sources)} points in computed.txt")

        print(f"Reading adjacency list...")
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            # Detect format: new format has source prefix, old is "s,n1,n2,..."
            has_tuples = '[' in first_line
            f.seek(0)

            for line in tqdm(f, desc=f"Processing edges", leave=False):
                line = line.strip()

                if has_tuples:
                    if spacev1b_sources is not None:
                        # Format: "[(neighbor, uncov_left), ...]" — no source prefix
                        source = spacev1b_sources[counter]
                        neighborhood = ast.literal_eval(line)
                    else:
                        # Format: "source [(neighbor, uncov_left), ...]"
                        space = line.index(' ')
                        source = int(line[:space])
                        neighborhood = ast.literal_eval(line[space+1:])  # list of (neighbor, uncov)
                    points = [source] + [nb for nb, _ in neighborhood]

                    # edges_to_99pct: first edge index where uncov_left <= 0.01 * n_nodes
                    threshold = 0.01 * n_nodes
                    edges_needed = len(neighborhood)   # default: needed all edges
                    for edge_idx, (_, uncov) in enumerate(neighborhood):
                        if uncov <= threshold:
                            edges_needed = edge_idx + 1
                            break
                    edges_to_99pct_sum += edges_needed
                else:
                    # Legacy format: "source,n1,n2,..."
                    points = [int(p.strip()) for p in line.split(',')]

                counter += 1

                chunk.append(points)

                if len(chunk) >= args.chunk_size:
                    process_chunk(chunk, outDeg, inDeg, n_nodes)
                    chunk = []
                    if counter % (args.chunk_size * 10) == 0:
                        outDeg.flush()
                        inDeg.flush()
                        gc.collect()

            if chunk:
                process_chunk(chunk, outDeg, inDeg, n_nodes)
                chunk = []

        avg_edges_to_99pct = round(edges_to_99pct_sum / counter, 2) if (has_tuples and counter) else None
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
            int(np.max(inDeg[:])),
            avg_edges_to_99pct,
        ])
        
        print(f"✓ Completed {dataset_name}-{metric}: {counter} points, avg out-degree: {np.round(np.mean(outDegNNZ), 2)}")
        print(f"  Saved degrees to: {outDeg_file} and {inDeg_file}")
        
        # Clean up
        del outDeg, inDeg
        gc.collect()
    
    # Merge new stats with existing stats
    new_stats_df = pd.DataFrame(stats, columns=['dataset', 'metric', 'dimensions', 'points computed', 'total points',
                                                  'mean out degree', 'median out degree', 'median in degree',
                                                  'min out degree', 'max out degree', 'min in degree', 'max in degree',
                                                  'avg edges to 99pct coverage'])
    
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
