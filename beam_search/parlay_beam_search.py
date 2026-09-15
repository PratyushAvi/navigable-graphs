#!/usr/bin/env python3
"""Run ParlayANN's recall harness on this project's adjacency lists.

Converts an adjacency list into a ParlayANN graph file (see
adj_to_parlay_graph.py), then hands it to ParlayANN's `neighbors` binary with
-graph_path, which loads the graph instead of building one. Search is therefore
the same code path used to evaluate Vamana -- the point of doing it this way
rather than with a separate Python beam search.

With --coverage the adjacency is truncated at each coverage level first, giving
the recall-vs-coverage sweep, one row per level.

Ground truth is required: without it ParlayANN writes its CSV header and no rows
at all. ann-benchmarks HDF5 files already carry `neighbors`/`distances` for the
test queries, so --hdf5 converts those directly and nothing is recomputed.

Example
-------
    python parlay_beam_search.py \\
        --adj-list  $RESULTS/adj-list-mnist-euclidean.txt \\
        --computed  $RESULTS/mnist-euclidean-computed.txt \\
        --base-fbin $DATA/mnist-784-euclidean/base.fbin \\
        --query-fbin $DATA/mnist-784-euclidean/query.fbin \\
        --hdf5      $DATA/mnist-784-euclidean.hdf5 \\
        --out-dir   $RESULTS/parlay_search/mnist \\
        --dataset   mnist --method robust-prune \\
        --coverage-range 99 100 0.25
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adj_to_parlay_graph import (                      # noqa: E402
    frange, load_adjacency, read_fbin_n, write_parlay_graph,
)

# The sweep driver already solved ground-truth conversion and result parsing;
# reuse it rather than keeping a second copy that can drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from gamma_sweep import (                              # noqa: E402
    groundtruth_from_hdf5, parlay_search,
)

SEARCH_COLUMNS = [
    "dataset", "metric", "method", "coverage", "k",
    "target recall", "beam width", "recall", "QPS",
    "mean seen", "tail seen", "mean expanded", "tail expanded",
    "queries", "pass", "nodes", "edges", "max degree", "search wall (s)",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adj-list", required=True)
    ap.add_argument("--computed",
                    help="computed-sources file for bare-neighbourhood adjacency")
    ap.add_argument("--base-fbin", required=True)
    ap.add_argument("--query-fbin", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metric", default="euclidean")
    ap.add_argument("--method", default="robust-prune",
                    help="label recorded in the CSV (default: robust-prune)")
    ap.add_argument("--hdf5",
                    help="ann-benchmarks HDF5; its neighbors/distances become "
                         "the ground truth")
    ap.add_argument("--gt-path", help="existing .gt file to use instead")
    ap.add_argument("--coverage", type=float, nargs="+",
                    help="coverage levels in percent, e.g. 99 99.5 100")
    ap.add_argument("--coverage-range", type=float, nargs=3,
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--k", type=int, default=10,
                    help="-k for the recall harness; ParlayANN sweeps its own "
                         "beam-width list, filtered to Q >= k (default: 10)")
    ap.add_argument("--vamana-bin",
                    default=str(Path(__file__).resolve().parent.parent
                                / "ParlayANN/algorithms/vamana/neighbors"),
                    help="ParlayANN neighbors binary used for searching")
    ap.add_argument("--R", type=int, default=32,
                    help="recorded in ParlayANN's output only; the graph is "
                         "loaded, not built")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--keep-graphs", action="store_true",
                    help="keep the converted graph files (default: keep them; "
                         "they are reused on a rerun)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    binary = Path(args.vamana_bin)
    if not binary.exists():
        ap.error(f"{binary} not found - build it with `make` in its directory")

    # --- ground truth ----------------------------------------------------
    gt_path = Path(args.gt_path) if args.gt_path else out_dir / f"{args.dataset}.gt"
    if not gt_path.exists():
        if not args.hdf5:
            ap.error(
                "no ground truth. Pass --hdf5 (ann-benchmarks files ship their "
                "own neighbors/distances) or --gt-path. Without it ParlayANN "
                "writes a header and no result rows.")
        groundtruth_from_hdf5(args.hdf5, gt_path, args.k)
    else:
        print(f"ground truth present: {gt_path}")

    # --- convert ---------------------------------------------------------
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
        print(f"coverage levels: {levels}")
        levels = [c / 100.0 for c in levels]

    adj, sources = load_adjacency(args.adj_list, n, args.computed, levels,
                                  args.limit)
    print(f"read {sources:,} neighbourhoods")
    if sources < n:
        print(f"NOTE: {n - sources:,} points have no neighbourhood and are "
              f"isolated; search starts at node 0, so they cannot be reached.")

    stem = Path(args.adj_list).stem
    graphs = []          # (coverage_label, path, edges, max_deg)
    if levels is None:
        p = out_dir / f"graph-{stem}"
        max_deg, total = write_parlay_graph(p, adj[0], n)
        print(f"wrote {p.name}  ({total:,} edges, max degree {max_deg})")
        graphs.append(("", p, total, max_deg))
    else:
        for li, c in enumerate(levels):
            p = out_dir / f"graph-{stem}-cov{c*100:g}"
            max_deg, total = write_parlay_graph(p, adj[li], n)
            print(f"wrote {p.name}  ({total:,} edges, max degree {max_deg})")
            graphs.append((round(c * 100, 6), p, total, max_deg))

    # --- search ----------------------------------------------------------
    csv_path = out_dir / f"parlay-search-{args.dataset}-{args.method}.csv"
    print(f"\n=== search (k={args.k}) ===")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SEARCH_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for cov, gpath, total, max_deg in graphs:
            label = f"coverage {cov}%" if cov != "" else "whole graph"
            print(f"[{label}]", flush=True)
            rows = parlay_search(
                binary, gpath, args.base_fbin, args.query_fbin, gt_path,
                out_dir / f"res-{gpath.name}.csv",
                args.k, args.R, args.L, args.alpha, verbose=args.verbose)
            base = {"dataset": args.dataset, "metric": args.metric,
                    "method": args.method, "coverage": cov,
                    "nodes": n, "edges": total, "max degree": max_deg}
            for row in rows:
                w.writerow({**base, **row})
            best = max(rows, key=lambda x: x["recall"])
            print(f"    {len(rows)} rows; best recall {best['recall']:.4f} "
                  f"needed Q={best['beam width']} "
                  f"(QPS {best['QPS']:.0f}, seen {best['mean seen']:.0f})",
                  flush=True)

    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
