import numpy as np
import cupy as cp
import h5py
from tqdm import tqdm
from cupyx.scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist as npcdist
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import rankdata
import psutil

### METHOD TO SAVE GRAPHS ###

def saveGraph(n, graph, outfile):
    print("Building adjacency matrix...")
    E = []
    for i in range(n):
        for j in range(1, len(graph[i])):
            E.append((graph[i][0], graph[i][j]))

    # Create adjacency matrix on GPU
    # adj = cp.zeros((n, n), dtype=cp.int16)
    # for (u, v) in E:
    #    adj[u][v] = 1

    # Transfer back to CPU and save
    # adj_cpu = cp.stack(adj.nonzero()).get()
    np.save(outfile, np.array(E))

    print(f"Saved adjacency matrix with {len(E)} edges. Shape: {adj_cpu.shape}")

########################
### GREEDY SET COVER ###

def buildSetsOfSource(permutation_matrix, source):
    """
    GPU-accelerated version - vectorized computation.
    
    Args:
        permutation_matrix: CuPy array (n, n) on GPU
        source: int, source vertex index
    
    Returns:
        CuPy array (n, n) on GPU
    """
    # Vectorized comparison: all rows compared to source row at once
    threshold = permutation_matrix[:, source][:, None]
    setsFromSourceVia = (permutation_matrix < threshold).astype(cp.uint16)
    
    return setsFromSourceVia

def greedySetCover(sets, permutation_matrix, source):
    """
    GPU-accelerated greedy set cover.
    
    Args:
        sets: CuPy array (n, n) on GPU
        permutation_matrix: CuPy array (n, n) on GPU
        source: int, source vertex index
    
    Returns:
        list of edge indices (transferred to CPU)
    """
    n = permutation_matrix.shape[0]
    
    # Initialize on GPU
    covered = cp.zeros(n, dtype=cp.bool_)
    covered[source] = True
    edges = []
    
    uncovered = n - 1
    sets_bool = sets.astype(cp.bool_)
    
    # while not cp.all(covered):            
    while uncovered > 0:
        # Compute scores on GPU (matrix-vector multiplication)
        scores = (~covered).astype(cp.int32) @ sets
        
        # Find argmax on GPU
        index = int(cp.argmax(scores))
        edges.append(index)
        
        # Update covered set
        newly_covered = sets_bool[:, index] & (~covered)
        uncovered -= int(cp.sum(newly_covered))
        covered |= newly_covered
    
    return edges

def buildSetCoverGraph(permutation_matrix):
    """
    Build graph using GPU-accelerated operations.
    
    Args:
        permutation_matrix: CuPy array (n, n) on GPU
    
    Returns:
        list of edge lists for each vertex
    """
    n = permutation_matrix.shape[0]
    edgeSet = []
    
    print("Building graph on GPU")

    for source in tqdm(range(n)):
        # Build sets for this source on GPU
        # print(f"{source} building sets", end='\r')
        sets = buildSetsOfSource(permutation_matrix, source)
        # print(f"{source} done building", end='\r')
        # Run greedy set cover on GPU
        edges = greedySetCover(sets, permutation_matrix, source)
        edgeSet.append(edges)
    
    return edgeSet

#######################

####################
### ROBUST PRUNE ###

def robustPrune(source, permutation_matrix):
    """
    GPU-optimized version of robustPrune using CuPy.
    
    Args:
        source: Source vertex index
        permutation_matrix: Distance matrix (cupy.ndarray on GPU)
    
    Returns:
        List of pruned edges in order
    """
    n = permutation_matrix.shape[0]
    
    # Get distances from source to all vertices
    dist_from_source = permutation_matrix[source].copy()
    
    # Create mask for active vertices (exclude source)
    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    
    edges = []
    
    while cp.any(active):
        # Find closest active vertex
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()
        
        # Add to edges and deactivate
        edges.append(waypoint)
        active[waypoint] = False
        
        # Vectorized pruning: remove vertices closer to waypoint than to source
        # For all active vertices, check if dist(waypoint, v) <= dist(source, v)
        prune_mask = (permutation_matrix[:, waypoint] <= permutation_matrix[:, source]) & active
        active[prune_mask] = False
    
    return edges

def memEfficientRobustPrune(source, dataset):
    n = dataset.shape[0]
    dist_from_source = cdist(dataset[source:source+1], dataset, metric='euclidean').flatten()

    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    
    edges = [source]

    while cp.any(active):
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()

        edges.append(waypoint)
        active[waypoint] = False
        
        # print(active.shape, dist_from_source.shape)

        prune_mask = (cdist(dataset, dataset[waypoint:waypoint+1], metric='euclidean').ravel() < dist_from_source) & active
        active[prune_mask] = False

    return edges

def angularRobustPrune(source, dataset):
    n = dataset.shape[0]

    cosine_distance = cdist(dataset[source:source+1], dataset, metric='cosine').flatten()
    dist_from_source = cp.arccos(1 - cosine_distance)
    
    active = cp.ones(n, dtype=cp.bool_)
    active[source] = False
    
    edges = [source]

    while cp.any(active):
        masked_dist = cp.where(active, dist_from_source, cp.inf)
        waypoint = cp.argmin(masked_dist).item()

        edges.append(waypoint)
        active[waypoint] = False
        
        # print(active.shape, dist_from_source.shape)

        prune_mask = (cp.arccos(1 - cdist(dataset, dataset[waypoint:waypoint+1], metric='cosine')).ravel() < dist_from_source) & active
        active[prune_mask] = False

    return edges

def jaccardRobustPrune(source, dataset):
    n = dataset.shape[0]

    dist_from_source = npcdist(dataset[source:source+1], dataset, metric='jaccard').flatten()
    
    active = np.ones(n, dtype=np.bool_)
    active[source] = False
    
    edges = [source]

    while cp.any(active):
        masked_dist = np.where(active, dist_from_source, np.inf)
        waypoint = np.argmin(masked_dist).item()

        edges.append(waypoint)
        active[waypoint] = False
        
        # print(active.shape, dist_from_source.shape)

        prune_mask = (npcdist(dataset, dataset[waypoint:waypoint+1], metric='jaccard').ravel() < dist_from_source) & active
        active[prune_mask] = False

    return edges


def memBuildRobustPruneGraph(dataset):
    n = dataset.shape[0]
    edgeSet = []

    for source in tqdm(range(n)):
        edges = memEfficientRobustPrune(source, dataset)
        edgeSet.append(edges)

    return edgeSet

def smallSampleBuildRobustPruneGraph(dataset, points):
    n = dataset.shape[0]
    edgeSet = []

    for source in points:
        edges = memEfficientRobustPrune(source, dataset)
        edgeSet.append(edges)

    return edgeSet

def buildRobustPruneGraph(permutation_matrix):
    """
    Build graph using GPU-accelerated operations.
    
    Args:
        permutation_matrix: CuPy array (n, n) on GPU
    
    Returns:
        list of edge lists for each vertex
    """
    n = permutation_matrix.shape[0]
    edgeSet = []
    
    print("Building graph on GPU")
    for source in tqdm(range(n)):
        edges = robustPrune(source, permutation_matrix)
        edgeSet.append(edges)
    
    return edgeSet

########################

################################
### HYBRID SET COVER METHODS ###

def greedySetCoverWithFriends(sets, friends, covered):
    n = sets.shape[0]
    edges = []
    
    # Pre-compute friend mask once
    fcover = cp.zeros(n, dtype=cp.bool_)
    fcover[friends] = True
    uncovered = len(friends) - 1
    
    while uncovered > 0:
        # Compute scores using matrix multiplication
        scores = (~covered).astype(cp.int32) @ sets
        
        index = int(cp.argmax(scores))
        edges.append(friends[index])
        
        # Update covered set
        newly_covered = cp.array(sets[:, index], dtype=cp.bool_)
        covered |= newly_covered
        uncovered -= int(cp.sum(newly_covered & fcover))
    
    return edges, covered

def buildHybridSetCover(dataset, k):
    n = dataset.shape[0]
    edgeSet = []
    
    # Pre-allocate distance array
    distanceFromSource = np.empty(n, dtype=np.float32)
    
    for source in tqdm(range(n)):
        covered = cp.zeros(n, dtype=cp.bool_)
        covered[source] = True  # Mark source as covered upfront
        
        # Compute all distances once
        distanceFromSource[:] = cdist(dataset, dataset[source:source+1], metric='euclidean').flatten()
        sourceEdges = []
        
        while int(cp.sum(covered)) < n:
            # Mask covered points
            masked_dist = distanceFromSource.copy()
            masked_dist[covered.get()] = np.inf
            
            # Get k nearest uncovered neighbors (excluding source)
            masked_dist[source] = np.inf
            friends = np.argpartition(masked_dist, min(k, n - int(cp.sum(covered)) - 1))[:k]
            friends = friends[masked_dist[friends] < np.inf]
            
            if len(friends) == 0:
                break
            
            # Always include source in friends
            friends = np.concatenate([[source], friends])
            
            # Compute permutation matrix only for friends
            friend_data = dataset[friends]
            distances = cdist(dataset, friend_data, metric='euclidean')
            permutation_matrix = cp.asarray(rankdata(distances, method='ordinal', axis=1), dtype=cp.uint16)
            source_idx = 0
                
            sets = buildSetsOfSource(permutation_matrix, source_idx)
            edges, covered = greedySetCoverWithFriends(sets, friends, covered)
            sourceEdges.extend(edges)
        
        edgeSet.append(sourceEdges)
    
    return edgeSet

def buildHybridSetCoverWithBetterFriends(dataset, k, v):
    n = dataset.shape[0]
    edgeSet = []

    for source in tqdm(range(n)):
        covered = cp.zeros(n, dtype=cp.bool_)
        covered[source] = True

        # Compute distances once per source
        distanceFromSource = cdist(dataset, dataset[source:source+1], metric='euclidean').ravel()
        sourceEdges = []

        while int(cp.sum(covered)) < n:
            uncovered_mask = ~covered.get()
            uncovered_mask[source] = False
            uncovered_indices = np.where(uncovered_mask)[0]

            if len(uncovered_indices) == 0:
                break

            # Vote efficiently
            voters = np.random.choice(uncovered_indices, size=min(v, len(uncovered_indices)), replace=False)
            voter_dists = cdist(dataset, dataset[voters], metric='euclidean')
            votes = cp.sum(voter_dists < distanceFromSource[:, None], axis=1)

            # Get top k friends from uncovered points
            possible_friends = min(k, len(uncovered_indices))
            uncovered_votes = votes[uncovered_mask]
            top_k_local = np.argpartition(uncovered_votes, -possible_friends)[-possible_friends:]
            friends = uncovered_indices[top_k_local]

            friends = np.concatenate([[source], friends])

            # Compute distances only for friends
            distances = cdist(dataset, dataset[friends], metric='euclidean')
            permutation_matrix = cp.asarray(rankdata(distances, method='ordinal', axis=1), dtype=cp.uint16)

            sets = buildSetsOfSource(permutation_matrix, 0)
            edges, covered = greedySetCoverWithFriends(sets, friends, covered)
            sourceEdges.extend(edges)

        edgeSet.append(sourceEdges)

    return edgeSet

def computeDistances(X, Y, metric):
    if metric in ['euclidean', 'jaccard']:
        return cdist(X, Y, metric=metric)
    elif metric == 'angular':
        return cp.arccos(cp.clip(1 - cdist(X, Y, metric='cosine'), -1, 1))
    else:
        return None

### CHUNKING FOR MEMORY EFFICIENT ROBUST PRUNE ###

def auto_chunk_size(dataset_shape, safety_factor=0.6):
    """Automatically determine optimal chunk size"""
    n, d = dataset_shape
    
    # Get available RAM
    available_ram = psutil.virtual_memory().available
    
    # Memory needed per row: 
    # - n * 8 bytes for distances (float64)
    # - n * ceil(log_2(n)/8) bytes for ranks (bits converted to bytes)
    bytes_for_ranks = n * np.ceil(np.log2(n) / 8)
    bytes_per_row = n * 8 + bytes_for_ranks
    
    # Calculate chunk size
    max_chunk = int((available_ram * safety_factor) / bytes_per_row)
    
    # Apply practical limits
    chunk_size = max(100, min(max_chunk, n, 10000))
    
    print(f"Dataset: {n} samples, {d} features")
    print(f"Available RAM: {available_ram / 1e9:.1f} GB")
    print(f"Bits per rank: {np.log2(n):.2f}")
    print(f"Bytes per rank: {np.ceil(np.log2(n) / 8):.0f}")
    print(f"Recommended chunk_size: {chunk_size}")
    
    return chunk_size

def checkNavigability(dist, graph):
    n = dist.shape[0]
    for i in tqdm(range(n)):
        # distances = np.array([np.linalg.norm(dataset[i] - dataset[e[1]]) for e in graph[i])
        for j in np.random.choice(n, 1000, replace=False):
            if j == i:
                continue
            distances = np.array([dist[j][e] for e in graph[i]])
            if not np.any(distances <= dist[i][j]):
                print("Found no greedy edge. G is not navigable.")
                print(i, j, dist[i][j], '\n', distances)
                return

    print("G appears to be navigable")
