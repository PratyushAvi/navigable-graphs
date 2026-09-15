#!/usr/bin/env python3
"""Convert this project's adjacency lists into ParlayANN graph files.

Lets ParlayANN's own recall harness evaluate robust-prune / set-cover graphs, so
they are measured by exactly the same search code as Vamana rather than by a
separate Python implementation.

Handles both adjacency layouts this project produces:

  "<source> [(nbr, uncov), ...]"   simulrun.py / Vamana-Runs.ipynb / gamma_sweep
  "[(nbr, uncov), ...]"            distributed_robust_prune.py, whose source ids
                                   live in a separate -computed.txt file
  "<source> [nbr, nbr, ...]"       already-merged graphs (graph_merge.py)

Tuple-form files carry the per-edge uncovered count, so --coverage truncates each
neighbourhood at a coverage level and writes one graph file per level. That is
the recall-vs-coverage sweep, done against ParlayANN's search.

ParlayANN graph format (see Graph in ParlayANN/algorithms/utils/graph.h):

    [uint32 n][uint32 max_deg][uint32 degrees[n]][uint32 edges, packed]

The edge block is ragged -- each node contributes exactly its own degree -- so
max_deg is a header field, not a stride.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional here
    def tqdm(it, **kw):
        return it


def read_fbin_n(path):
    """Point count from an fbin header, without reading the vectors."""
    with open(path, "rb") as f:
        n, _dims = np.fromfile(f, dtype=np.int32, count=2)
    return int(n)


def parse_line(line, computed_source=None):
    """Return (source, neighborhood). `neighborhood` may be tuples or bare ids."""
    line = line.strip()
    if not line:
        return None, None
    if line.startswith("["):
        if computed_source is None:
            raise ValueError(
                "line has no source prefix; pass --computed with the matching "
                "computed-sources file (distributed_robust_prune.py writes one)")
        return computed_source, ast.literal_eval(line)
    sp = line.index(" ")
    return int(line[:sp]), ast.literal_eval(line[sp + 1:])


def load_adjacency(adj_path, n, computed_path=None, coverages=None,
                   limit=None):
    """Read an adjacency file into one neighbour list per coverage level.

    With `coverages` None the neighbourhoods are taken whole. Otherwise each is
    truncated at every requested level, and the return value has one adjacency
    per level, in the same order.

    The truncation rule is the one in beam_search.load_graphs: `uncov` is the
    count still uncovered AFTER an edge, so to reach coverage c every edge must
    be kept up to and including the one that first brings the count to or below
    n*(1-c). Keying off the count BEFORE each edge does that; keying off the
    stored (post-edge) count would drop exactly that crossing edge and leave
    some sources with too few, or zero, edges.
    """
    levels = [None] if coverages is None else list(coverages)
    adj = [[[] for _ in range(n)] for _ in levels]

    computed = None
    if computed_path:
        with open(computed_path) as cf:
            computed = [int(x.strip()) for x in cf if x.strip()]

    seen, tuple_form = set(), None
    with open(adj_path) as f:
        for i, line in enumerate(tqdm(f, desc=Path(adj_path).name)):
            if limit is not None and i >= limit:
                break
            src = computed[i] if computed is not None and i < len(computed) else None
            source, nbrs = parse_line(line, src)
            if source is None:
                continue
            if not 0 <= source < n:
                raise ValueError(f"{adj_path}:{i+1}: source {source} outside "
                                 f"dataset of {n} points")
            if source in seen:
                raise ValueError(f"{adj_path}:{i+1}: duplicate source {source}")
            seen.add(source)

            if nbrs:
                is_tuple = isinstance(nbrs[0], (tuple, list))
                if tuple_form is None:
                    tuple_form = is_tuple
                elif tuple_form != is_tuple:
                    raise ValueError(f"{adj_path}:{i+1}: mixed line formats")

            if coverages is None:
                adj[0][source] = [int(t[0]) if tuple_form else int(t) for t in nbrs]
                continue

            if not tuple_form:
                raise ValueError(
                    f"{adj_path} has no (neighbor, uncov) tuples, so --coverage "
                    f"cannot truncate it. Convert it whole (drop --coverage).")
            prev_uncov = n - 1        # only the source is covered before any edge
            for neighbor, uncov in nbrs:
                for li, c in enumerate(levels):
                    if prev_uncov > n * (1 - c):
                        adj[li][source].append(int(neighbor))
                prev_uncov = uncov

    if not seen:
        raise ValueError(f"{adj_path} produced no neighbourhoods")
    return adj, len(seen)


def write_parlay_graph(path, adj, n):
    """Write one adjacency (list of neighbour lists) as a ParlayANN graph file."""
    degrees = np.fromiter((len(a) for a in adj), dtype=np.uint32, count=n)
    max_deg = int(degrees.max()) if n else 0
    total = int(degrees.sum())
    edges = np.fromiter((v for a in adj for v in a), dtype=np.uint32, count=total)
    with open(path, "wb") as f:
        np.array([n, max_deg], dtype=np.uint32).tofile(f)
        degrees.tofile(f)
        edges.tofile(f)
    return max_deg, total


def frange(lo, hi, step):
    out, k = [], 0
    while True:
        v = round(lo + k * step, 10)
        if v > hi + 1e-9:
            break
        out.append(round(v, 6))
        k += 1
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adj-list", required=True,
                    help="adjacency file to convert")
    ap.add_argument("--computed",
                    help="computed-sources file, for adjacency files whose lines "
                         "are bare neighbourhoods (distributed_robust_prune.py)")
    ap.add_argument("--base-fbin", required=True,
                    help="base.fbin for this dataset; supplies the point count")
    ap.add_argument("--out", required=True,
                    help="output graph file, or its prefix when --coverage is given")
    ap.add_argument("--coverage", type=float, nargs="+",
                    help="coverage levels as percentages, e.g. 99 99.5 100. One "
                         "graph file is written per level, suffixed -cov<level>")
    ap.add_argument("--coverage-range", type=float, nargs=3,
                    metavar=("MIN", "MAX", "STEP"),
                    help="coverage levels as a grid instead of a list")
    ap.add_argument("--limit", type=int, help="stop after this many lines")
    args = ap.parse_args()

    n = read_fbin_n(args.base_fbin)
    print(f"dataset: {n:,} points")

    levels = None
    if args.coverage_range:
        levels = frange(*args.coverage_range)
    elif args.coverage:
        levels = list(args.coverage)
    if levels is not None:
        bad = [c for c in levels if not 0 < c <= 100]
        if bad:
            ap.error(f"coverage levels must be in (0, 100]: {bad}")
        levels = [c / 100.0 for c in levels]
        print(f"coverage levels: {[round(c*100, 4) for c in levels]}")

    adj, sources = load_adjacency(args.adj_list, n, args.computed, levels,
                                  args.limit)
    print(f"read {sources:,} neighbourhoods")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    if levels is None:
        max_deg, total = write_parlay_graph(out, adj[0], n)
        print(f"wrote {out}  ({total:,} edges, max degree {max_deg})")
        written.append(out)
    else:
        for li, c in enumerate(levels):
            p = out.with_name(f"{out.name}-cov{c*100:g}")
            max_deg, total = write_parlay_graph(p, adj[li], n)
            print(f"wrote {p}  ({total:,} edges, max degree {max_deg})")
            written.append(p)

    if sources < n:
        print(f"\nNOTE: {n - sources:,} of {n:,} points had no neighbourhood in "
              f"the adjacency file and are isolated in the graph. Beam search "
              f"starts at node 0, so anything unreachable will not be found.")
    return written


if __name__ == "__main__":
    main()
