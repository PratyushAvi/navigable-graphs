"""Convert a ParlayANN CSR graph file into the (neighbor, uncov) adj-list format.

ParlayANN stores only edges, so the uncovered counts are recomputed here using the
same alpha-reachability rule as distributed_robust_prune.py: waypoint w covers p
when d(w, p) * alpha < d(s, p). Distances are squared euclidean, so the test scales
by alpha^2, and p stays uncovered while d2(s, p) <= d2(w, p) * alpha_sq.

Edges are replayed in the order ParlayANN stored them, and each edge records the
uncovered count *after* it takes effect, matching the writer in
distributed_robust_prune.writeNeighborhood.
"""

import argparse

import h5py
import numpy as np


def read_csr(path):
    """Read a ParlayANN graph file into (list of neighbor arrays, max_deg)."""
    with open(path, "rb") as f:
        n, max_deg = np.fromfile(f, dtype=np.uint32, count=2)
        sizes = np.fromfile(f, dtype=np.uint32, count=int(n))
        edges = np.fromfile(f, dtype=np.uint32)
    offsets = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])
    return [edges[offsets[i]:offsets[i + 1]] for i in range(int(n))], int(max_deg)


def read_fbin(path):
    with open(path, "rb") as f:
        n, dims = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32).reshape(n, dims)


def load_vectors(path):
    if path.endswith(".hdf5"):
        with h5py.File(path, "r") as f:
            return np.ascontiguousarray(f["train"][:], dtype=np.float32)
    return read_fbin(path)


def sq_dists(vectors, norms, point):
    """Squared euclidean distance from `point` to every row of `vectors`."""
    d = norms - 2.0 * (vectors @ point)
    return d + point @ point


def neighborhood_with_uncov(source, neighbors, vectors, norms, alpha_sq):
    """Replay one node's edges, recording uncovered count after each edge."""
    d_source = sq_dists(vectors, norms, vectors[source])
    uncov = np.arange(len(vectors), dtype=np.int32)
    uncov = uncov[uncov != source]

    out = []
    for w in neighbors:
        w = int(w)
        d_w = sq_dists(vectors, norms, vectors[w])
        uncov = uncov[d_source[uncov] <= d_w[uncov] * alpha_sq]
        uncov = uncov[uncov != w]
        out.append((w, int(len(uncov))))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", required=True, help="ParlayANN CSR graph file")
    p.add_argument("--vectors", required=True, help="base .fbin or source .hdf5")
    p.add_argument("--out", required=True, help="adj-list output path")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--limit", type=int, help="only convert the first N nodes")
    args = p.parse_args()

    if args.alpha < 1.0:
        raise ValueError(f"alpha must be >= 1, got {args.alpha}")
    alpha_sq = args.alpha ** 2

    graph, _ = read_csr(args.graph)
    vectors = load_vectors(args.vectors)
    norms = np.einsum("ij,ij->i", vectors, vectors)

    total = min(len(graph), args.limit) if args.limit else len(graph)
    with open(args.out, "w") as out:
        for i in range(total):
            nbrs = neighborhood_with_uncov(i, graph[i], vectors, norms, alpha_sq)
            out.write(f"{i} {nbrs}\n")
            if (i + 1) % 1000 == 0:
                print(f"{i + 1}/{total} nodes", flush=True)


if __name__ == "__main__":
    main()
