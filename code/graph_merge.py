"""Merge subgraphs built by simulrun.py into a single navigable graph.

Input is a manifest file listing the dataset first, then one adjacency file per
subgraph:

    /path/to/dataset.hdf5
    /path/to/adj-list-sift-0_250-euclidean.txt
    /path/to/adj-list-sift-250_500-euclidean.txt
    ...

Each point keeps the neighbors its own subgraph gave it, and additionally gains
one edge to its nearest point in every *other* subgraph. The result is written
next to the first adjacency file with MERGE_BUILD in the name.

Cross-links are computed with a chunked exact search (no index), so the cost is
O(|shard_i| * |shard_j| * d) per ordered pair. That is fine for the shard sizes
simulrun.py produces but is not a billion-scale tool.
"""

import argparse
import ast
import os

import h5py
import numpy as np
from tqdm import tqdm


def parse_manifest(path):
    """Return (dataset_path, [adj_path, ...]) from the manifest file.

    Blank lines and '#' comments are ignored so manifests can be annotated.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f]
    entries = [ln for ln in lines if ln and not ln.startswith('#')]

    if len(entries) < 2:
        raise ValueError(
            f"manifest must list a dataset and at least one adjacency file, "
            f"got {len(entries)} entr{'y' if len(entries) == 1 else 'ies'}"
        )

    dataset_path, adj_paths = entries[0], entries[1:]
    for p in [dataset_path] + adj_paths:
        if not os.path.exists(p):
            raise ValueError(f"path listed in manifest does not exist: {p}")

    dupes = {p for p in adj_paths if adj_paths.count(p) > 1}
    if dupes:
        raise ValueError(f"adjacency file listed more than once: {sorted(dupes)}")

    return dataset_path, adj_paths


def read_adjacency(path):
    """Read one simulrun.py adjacency file.

    Lines are "<source_id> [(neighbor, uncov), ...]" — the tuple form written by
    the euclidean branch. Returns {source_id: [neighbor_id, ...]}, dropping the
    uncov counts, which describe coverage during construction and carry no
    meaning once subgraphs are merged.
    """
    adjacency = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                space = line.index(' ')
                source = int(line[:space])
                neighborhood = ast.literal_eval(line[space + 1:])
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"{os.path.basename(path)}:{lineno}: cannot parse line "
                    f"(expected '<source> [(neighbor, uncov), ...]'): {e}"
                )
            if neighborhood and not isinstance(neighborhood[0], tuple):
                raise ValueError(
                    f"{os.path.basename(path)}:{lineno}: legacy format without "
                    f"uncov tuples is not supported; rebuild with simulrun.py"
                )
            if source in adjacency:
                raise ValueError(
                    f"{os.path.basename(path)}:{lineno}: duplicate source {source}"
                )
            adjacency[source] = [int(nb) for nb, _ in neighborhood]
    return adjacency


def nearest_across(sources, targets, vectors, chunk_size):
    """For each id in `sources`, find the nearest id in `targets`.

    Exact search via chunked squared-euclidean distance. `sources` and `targets`
    are arrays of global dataset indices; returns an array parallel to `sources`
    holding the winning global target index for each.

    Chunking is over sources so peak memory is chunk_size * len(targets) floats
    regardless of how large the source shard is.
    """
    tgt_vecs = vectors[targets]                                  # (m, d)
    tgt_sq = np.einsum('ij,ij->i', tgt_vecs, tgt_vecs)           # (m,)

    best = np.empty(len(sources), dtype=np.int64)
    for i in range(0, len(sources), chunk_size):
        src_ids = sources[i:i + chunk_size]
        src_vecs = vectors[src_ids]                              # (k, d)
        # ||s - t||^2 = ||s||^2 - 2 s·t + ||t||^2; the ||s||^2 term is constant
        # across targets so it cannot change the argmin and is omitted.
        d = tgt_sq[None, :] - 2.0 * (src_vecs @ tgt_vecs.T)      # (k, m)
        best[i:i + chunk_size] = targets[np.argmin(d, axis=1)]
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Merge simulrun.py subgraphs by linking each point to its "
                    "nearest neighbor in every other subgraph.")
    parser.add_argument('manifest',
                        help="Text file listing the dataset, then one adjacency "
                             "file per subgraph (one path per line).")
    parser.add_argument('--hdf5_key', default='train',
                        help="Key inside the HDF5 dataset file (default: train)")
    parser.add_argument('--chunk_size', type=int, default=4096,
                        help="Sources per distance chunk. Peak memory is roughly "
                             "chunk_size * (largest shard) * 4 bytes (default: 4096)")
    parser.add_argument('--output', default=None,
                        help="Output path. Defaults to the first adjacency file "
                             "with MERGE_BUILD inserted before the extension.")
    args = parser.parse_args()

    try:
        dataset_path, adj_paths = parse_manifest(args.manifest)
    except ValueError as e:
        parser.error(str(e))

    print(f"Dataset:   {dataset_path}")
    print(f"Subgraphs: {len(adj_paths)}")

    if len(adj_paths) == 1:
        print("Only one subgraph listed — there are no other subgraphs to link "
              "to, so the output would just copy the input. Nothing to do.")
        return

    # --- Load subgraphs ---
    subgraphs = []
    for p in adj_paths:
        try:
            adj = read_adjacency(p)
        except ValueError as e:
            parser.error(str(e))
        if not adj:
            parser.error(f"adjacency file is empty: {p}")
        print(f"  {os.path.basename(p)}: {len(adj)} sources")
        subgraphs.append(adj)

    # A point owned by two subgraphs would get contradictory neighbor sets and
    # ambiguous cross-links, so refuse rather than silently picking one.
    seen = {}
    for i, adj in enumerate(subgraphs):
        for src in adj:
            if src in seen:
                parser.error(
                    f"source {src} appears in both {os.path.basename(adj_paths[seen[src]])} "
                    f"and {os.path.basename(adj_paths[i])}; subgraphs must be disjoint")
            seen[src] = i

    shard_ids = [np.fromiter(sorted(adj), dtype=np.int64, count=len(adj))
                 for adj in subgraphs]
    total_sources = sum(len(s) for s in shard_ids)
    print(f"Total: {total_sources} sources across {len(subgraphs)} disjoint subgraphs")

    # --- Load vectors ---
    with h5py.File(dataset_path, 'r') as f:
        if args.hdf5_key not in f:
            parser.error(f"key {args.hdf5_key!r} not in {dataset_path}; "
                         f"available: {list(f.keys())}")
        vectors = np.asarray(f[args.hdf5_key], dtype=np.float32)
    n_points, dim = vectors.shape
    print(f"Vectors: {n_points:,} x {dim}")

    out_of_range = [int(s) for ids in shard_ids for s in ids[ids >= n_points][:1]]
    if out_of_range:
        parser.error(f"source id {out_of_range[0]} is outside the dataset "
                     f"({n_points} points) — do the adjacency files match this dataset?")

    # --- Cross-subgraph links ---
    # For every ordered pair (i, j), every point of shard i gains an edge to its
    # nearest point in shard j. Each pair is one chunked exact search.
    cross = {}
    pairs = [(i, j) for i in range(len(subgraphs)) for j in range(len(subgraphs)) if i != j]
    for i, j in tqdm(pairs, desc="Linking subgraph pairs"):
        nearest = nearest_across(shard_ids[i], shard_ids[j], vectors, args.chunk_size)
        for src, tgt in zip(shard_ids[i], nearest):
            cross.setdefault(int(src), []).append(int(tgt))

    # --- Write merged adjacency ---
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(adj_paths[0])
        out_path = f"{base}-MERGE_BUILD{ext}"

    n_edges = 0
    n_added = 0
    with open(out_path, 'w') as out:
        for adj in subgraphs:
            for source in sorted(adj):
                original = adj[source]
                # Keep the subgraph's own edges, then append cross-links that
                # aren't already present. Order is preserved so the original
                # neighborhood stays a prefix of the merged one.
                merged = list(original)
                known = set(original)
                for tgt in cross.get(source, []):
                    if tgt not in known and tgt != source:
                        merged.append(tgt)
                        known.add(tgt)
                n_added += len(merged) - len(original)
                n_edges += len(merged)
                out.write(f"{source} {merged}\n")

    print(f"\nWrote {total_sources} sources, {n_edges} edges "
          f"({n_added} new cross-subgraph edges) → {out_path}")


if __name__ == '__main__':
    main()
