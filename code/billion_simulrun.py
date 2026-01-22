import numpy as np
import argparse
from tqdm import tqdm
from time import time
import psutil

def main():
    DATASET = "spacev1b" # Get dataset name from argument
    METRIC = 'euclidean'

    print("Building graph on", DATASET, flush=True)
    SAVEPATH = "/scratch/pa2439/Projects/ANN-Search/navigable_graph_results/results"
    # BINARY_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors.mmap"
    BINARY_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_int.mmap"
    SQ_NORMS_FILE = "/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/sq_vector_norms.mmap"
    dataset_shape = np.load("/scratch/pa2439/Projects/ANN-Search/datasets/SPACEV1B/vectors_shape.npy")
    MEMPOINT =  10 ** 8

    # X = np.memmap(BINARY_FILE, mode='r', dtype='float32', shape=dataset_shape)
    X = np.memmap(BINARY_FILE, mode='r', dtype='int8', shape=dataset_shape)
    print("Loaded vectors:", X.shape, flush=True)

    # sq_norms = np.memmap(SQ_NORMS_FILE, mode='r', dtype='float32', shape=X.shape[0])
    # print("Loaded squared norms", flush=True)

    metric = 'euclidean'

    # assume no other program is trying to edges simultaneously
    with open(f"{SAVEPATH}/{DATASET}-{metric}-computed.txt", 'a+') as f:
        f.seek(0)
        completed = set([int(line.strip()) for line in f if line.strip()])
    
    all_sources = np.arange(X.shape[0], dtype=np.uint32)
    np.random.shuffle(all_sources)

    for source in all_sources[:1]:
        if source not in completed:
            
            ### compute neighborhood of source ###

            edges = [source]

            active = np.zeros(X.shape[0], dtype=np.uint8)
            active[source] = 0

            
            print(f"Computing pairwise distances from {source}", flush=True)
            start = time()
            dist_from_source = -2 * (X @ X[source]) + (X[source] @ X[source])
            # dist_from_source = -2 * (X @ X[source]) + sq_norms[source]
            print(f"Took {time() - start} seconds to compute pairwise distances", flush=True)

            start = time()
            while np.sum(active) > MEMPOINT:
                matvectime = time()
                masked_dist = np.where(active > 0, dist_from_source, np.inf)
                waypoint = np.argmin(masked_dist)
                active[waypoint] = 0
                edges.append(waypoint)

                dist_from_waypoint = -2 * (X @ X[waypoint]) + (X[waypoint] @ X[waypoint])
                active[dist_from_waypoint < dist_from_source] = 0
                print(f"Degree: {len(edges) - 1}, Left: {np.sum(active)}, Took: {time() - matvectime}", flush=True)
            
            # active_indices = active[active > 0]
            # X_inmem = X[active_indices]
            # dist_inmem = dist_from_source[active_indices]
            # active_inmem = active[active_indices]
            # print(f"moving into memory..., {len(active_indices)}, {active_inmem.shape}, {X_inmem.shape}, {active_inmem[0]}", flush=True)

            # while np.sum(active_inmem) > 0:
            #     matvectime = time()
            #     masked_dist = np.where(active_inmem > 0, dist_inmem, np.inf)
            #     waypoint = np.argmin(masked_dist)
            #     active_inmem[waypoint] = 0
            #     edges.append(waypoint)

            #     dist_from_waypoint = -2 * (X_inmem @ X_inmem[waypoint]) + (X_inmem[waypoint] @ X_inmem[waypoint])
            #     active_inmem[dist_from_waypoint < dist_inmem] = 0
            #     print(f"Degree: {len(edges) - 1}, Left: {np.sum(active_inmem)}, Took: {time() - matvectime}", flush=True)

            ######################################

            print(f"Took {time()-start} seconds to compute neighborhood", flush=True)

            with open(f"{SAVEPATH}/adj-list-{DATASET}-{metric}.txt", 'a') as adj:
                # FIX: Added closing parenthesis and newline character
                adj.write(','.join([str(e) for e in edges]) + '\n')
            
            with open(f"{SAVEPATH}/{DATASET}-{metric}-computed.txt", 'a') as f:
                f.write(f"{source}\n")
        
    print(f"Done with {DATASET}")


def compute_sq_norms(X_memmap, memory_factor=0.8):
    """Computes the squared L2-norm for every vector in a large memmap."""
    
    N, D = X_memmap.shape
    
    # Get available RAM and determine memory limit for the chunk
    available_ram_bytes = psutil.virtual_memory().available
    chunk_ram_limit = available_ram_bytes * memory_factor
    
    # Calculate adaptive chunk size
    bytes_per_vector = D * (X_memmap.dtype.itemsize + np.dtype(np.float32).itemsize)
    max_chunk_size = int(chunk_ram_limit / bytes_per_vector)
    chunk_size = max(1, min(max_chunk_size, N))

    print(f"Total available RAM: {available_ram_bytes / 1024**3:.2f} GB")
    print(f"Chunk limit ({memory_factor*100:.0f}%): {chunk_ram_limit / 1024**3:.2f} GB")
    print(f"Using chunk size: {chunk_size} vectors")

    # Initialize result array
    sq_norms = np.empty(N, dtype=np.int16)

    # Process data in chunks
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        X_chunk = X_memmap[start:end] 
        
        # Vectorized squaring and summation
        sq_norms_chunk = np.sum(X_chunk * X_chunk, axis=1, dtype=np.int16)
        
        sq_norms[start:end] = sq_norms_chunk

    return sq_norms

def compute_inner_products(X_memmap, query_vector, memory_factor=0.8):
    """Computes the inner product of a query vector against every vector in a large memmap."""
    
    N, D = X_memmap.shape
    
    if query_vector.shape != (D,):
        raise ValueError(f"Query vector must be {D}-dimensional, but got {query_vector.shape}")
    
    q = query_vector.astype(np.float32)

    # Determine Memory Limit
    available_ram_bytes = psutil.virtual_memory().available
    chunk_ram_limit = available_ram_bytes * memory_factor
    
    # Calculate adaptive chunk size
    bytes_per_vector = D * (X_memmap.dtype.itemsize + np.dtype(np.float32).itemsize)
    max_chunk_size = int(chunk_ram_limit / bytes_per_vector)
    chunk_size = max(1, min(max_chunk_size, N))

    print(f"Total available RAM: {available_ram_bytes / 1024**3:.2f} GB", flush=True)
    print(f"Chunk limit ({memory_factor*100:.0f}%): {chunk_ram_limit / 1024**3:.2f} GB", flush=True)
    print(f"Using chunk size: {chunk_size} vectors", flush=True)

    # Initialize result array
    inner_products = np.empty(N, dtype=np.float32)

    # Process data in chunks
    for start in tqdm(range(0, N, chunk_size)):
        end = min(start + chunk_size, N)

        X_chunk = X_memmap[start:end] 
        
        # Convert to float32 to enable BLAS-optimized matrix multiplication
        X_float = X_chunk.astype(np.float32)

        # Perform the Matrix-Vector (GEMV) multiplication: X_chunk * q
        products_chunk = X_float @ q
        
        inner_products[start:end] = products_chunk
    
    return inner_products

if __name__ == "__main__":
    main()
