#!/usr/bin/env python3
"""Extract the edge-vs-coverage curve for the single source with the largest
out-degree in an adjacency-list file.

The per-dataset stats CSVs only hold aggregates (mean/median/min/max over all
sources), so they can't show what any individual source does. This script scans
one adj-list, finds the source with the most edges, and writes its own curve:

    edges, points covered, coverage frac

Useful for diagnosing sudden drops/kinks in the aggregate curves (e.g. bigann,
yandex_deep) that come from one or a few high-degree outliers.

Parsing mirrors edge_to_coverage_analysis.py:
  * Each adj-list tuple is (neighbor, uncov_after_edge). Points covered after
    the first k edges is total_points - uncov on the k-th tuple. (Older files
    store pre-edge uncov; run fix_adj_list_uncov.py to migrate them first.)
  * Line formats:
      - bare neighborhood:  [(n0, u0), ...]           (robust-prune; needs
                                                        --computed for source ids)
      - source-prefixed:    "<source_id> [(n0, u0), ...]"  (set-cover / legacy)

Usage:
    python extract_max_degree_curve.py \
        --adj-list path/to/adj-list-bigann-euclidean.txt \
        --computed path/to/bigann-euclidean-computed.txt \
        --total-points 1000000000 \
        --out max_degree_curve_bigann.csv
"""
import argparse
import ast
import csv
import os
import sys


def iter_sources(adj_list_path, computed_path):
    """Yield (source_id, neighborhood) for each line.

    source_id comes from the computed-sources file (bare lines) or the line
    prefix (source-prefixed lines). If neither is available, source_id is the
    0-based line index.
    """
    computed = None
    if computed_path is not None:
        with open(computed_path) as cf:
            computed = [int(p.strip()) for p in cf if p.strip()]

    with open(adj_list_path) as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                neighborhood = ast.literal_eval(line)
                if computed is not None:
                    if idx >= len(computed):
                        break
                    src = computed[idx]
                else:
                    src = idx
            else:
                space = line.index(" ")
                src = int(line[:space])
                neighborhood = ast.literal_eval(line[space + 1:])
            yield src, neighborhood
            idx += 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adj-list", required=True, help="adjacency-list file to scan")
    p.add_argument("--computed", default=None,
                   help="computed-sources file (source ids, one per line) for "
                        "bare-neighborhood adj-lists; omit for source-prefixed lines")
    p.add_argument("--total-points", type=int, required=True,
                   help="universe size coverage is measured against (n_nodes)")
    p.add_argument("--out", default=None,
                   help="output CSV path (default: max_degree_curve_<adjname>.csv "
                        "next to the adj-list)")
    args = p.parse_args()

    if not os.path.exists(args.adj_list):
        p.error(f"--adj-list not found: {args.adj_list}")
    if args.computed is not None and not os.path.exists(args.computed):
        p.error(f"--computed not found: {args.computed}")

    n_nodes = args.total_points

    best_src = None
    best_deg = -1
    best_neigh = None
    n_seen = 0
    for src, neighborhood in iter_sources(args.adj_list, args.computed):
        n_seen += 1
        deg = len(neighborhood)
        if deg > best_deg:
            best_deg = deg
            best_src = src
            best_neigh = neighborhood

    if best_neigh is None:
        print("No sources found.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanned {n_seen} sources. Max out-degree = {best_deg} "
          f"(source id {best_src}).")

    out_path = args.out or os.path.join(
        os.path.dirname(args.adj_list) or ".",
        "max_degree_curve_" + os.path.basename(args.adj_list).replace(".txt", ".csv"),
    )

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "total points", "edges",
                    "points covered", "coverage frac"])
        for j, (_neighbor, uncov) in enumerate(best_neigh):
            covered = n_nodes - uncov
            w.writerow([best_src, n_nodes, j + 1, covered, covered / n_nodes])

    print(f"Wrote {best_deg} rows -> {out_path}")


if __name__ == "__main__":
    main()
