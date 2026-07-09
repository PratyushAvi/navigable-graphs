#!/usr/bin/env python3
"""Diagnose sharp drops in the aggregate edge-vs-coverage curve.

A steep downward plunge in log(1 - coverage) vs edges means the uncovered
fraction collapses over a narrow range of edge counts. The usual cause is a
*saturation cliff*: many sources reach full coverage (uncov -> 0) at nearly the
same out-degree, then get clamped at coverage=1.0 for all higher edge counts, so
the mean's (1 - coverage) collapses right where they pile up.

This script scans one adj-list and writes two diagnostic CSVs:

  * <out>_active.csv:  per edge count k, how many sources are still "active"
        (out-degree > k, i.e. not yet saturated) and the mean coverage over
        ONLY those active sources vs. over ALL sources (clamped). A cliff in
        the ALL curve that is absent from the ACTIVE-only curve confirms the
        drop is a clamping/saturation artifact, not a per-source geometric jump.

  * <out>_degree_hist.csv:  the out-degree distribution (histogram). A tightly
        peaked distribution produces a sharp cliff; a heavy tail produces a
        gradual approach.

Parsing mirrors edge_to_coverage_analysis.py: tuples are (neighbor,
uncov_after_edge); points covered after the first k edges is
total_points - uncov on the k-th tuple.

Usage:
    python diagnose_coverage_drop.py \
        --adj-list .../adj-list-bigann-euclidean.txt \
        --computed .../bigann-euclidean-computed.txt \
        --total-points 1000000000 \
        --out ../edge_to_coverage/diag_bigann
"""
import argparse
import ast
import csv
import os
import sys

import numpy as np


def iter_neighborhoods(adj_list_path, computed_path):
    """Yield each source's neighborhood list (bare or source-prefixed lines)."""
    have_computed = computed_path is not None
    n_computed = None
    if have_computed:
        with open(computed_path) as cf:
            n_computed = sum(1 for p in cf if p.strip())

    with open(adj_list_path) as f:
        idx = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("["):
                if have_computed and idx >= n_computed:
                    break
                neighborhood = ast.literal_eval(line)
            else:
                space = line.index(" ")
                neighborhood = ast.literal_eval(line[space + 1:])
            yield neighborhood
            idx += 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adj-list", required=True)
    p.add_argument("--computed", default=None)
    p.add_argument("--total-points", type=int, required=True)
    p.add_argument("--out", required=True,
                   help="output prefix; writes <out>_active.csv and "
                        "<out>_degree_hist.csv")
    p.add_argument("--max-edges", type=int, default=None,
                   help="cap the per-edge sweep (default: largest out-degree seen)")
    args = p.parse_args()

    if not os.path.exists(args.adj_list):
        p.error(f"--adj-list not found: {args.adj_list}")
    if args.computed is not None and not os.path.exists(args.computed):
        p.error(f"--computed not found: {args.computed}")

    n_nodes = args.total_points

    # Store each source's per-edge covered-count curve and its degree.
    degrees = []
    per_source_cov = []  # list of np.int64 arrays (covered after first j+1 edges)
    for neighborhood in iter_neighborhoods(args.adj_list, args.computed):
        deg = len(neighborhood)
        degrees.append(deg)
        cov = np.fromiter((n_nodes - u for (_n, u) in neighborhood),
                          dtype=np.int64, count=deg)
        per_source_cov.append(cov)

    n_sources = len(degrees)
    if n_sources == 0:
        print("No sources found.", file=sys.stderr)
        sys.exit(1)
    degrees = np.array(degrees, dtype=np.int64)
    max_deg = int(degrees.max())
    print(f"Scanned {n_sources} sources. Out-degree: min={degrees.min()} "
          f"median={int(np.median(degrees))} max={max_deg}")

    max_edges = args.max_edges or max_deg

    # --- active.csv: active count + mean coverage (active-only vs all) per k ---
    active_path = f"{args.out}_active.csv"
    with open(active_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edges", "n_sources", "n_active", "frac_active",
                    "mean cov frac (all, clamped)", "mean cov frac (active only)"])
        for k in range(1, max_edges + 1):
            idx = k - 1
            covered_all = np.empty(n_sources, dtype=np.float64)
            active_mask = degrees > k  # source still has edges beyond k
            active_covered = []
            for i, cov in enumerate(per_source_cov):
                d = len(cov)
                # clamp: a source of degree d < k saturates at its last value
                covered_all[i] = cov[min(idx, d - 1)]
                if d > k:
                    active_covered.append(cov[idx])
            mean_all = covered_all.mean() / n_nodes
            mean_active = (np.mean(active_covered) / n_nodes
                           if active_covered else float("nan"))
            w.writerow([k, n_sources, int(active_mask.sum()),
                        active_mask.mean(), mean_all, mean_active])
    print(f"Wrote {active_path}")

    # --- degree_hist.csv: out-degree distribution ---
    hist_path = f"{args.out}_degree_hist.csv"
    counts = np.bincount(degrees, minlength=max_deg + 1)
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["out_degree", "n_sources"])
        for d in range(1, max_deg + 1):
            if counts[d]:
                w.writerow([d, int(counts[d])])
    print(f"Wrote {hist_path}")

    # --- saturation_deciles.csv: edge count at which each decile of sources is
    # fully covered. "Saturated" = final covered count == n_nodes, at which point
    # the source's out-degree IS the edge count where it reached full coverage
    # (that is why greedy pruning stopped). Sources that never reach full
    # coverage (residual uncov, e.g. duplicate points) have no finite saturation
    # edge and are excluded from the percentiles; their share is reported too.
    final_covered = np.array([c[-1] if len(c) else 0 for c in per_source_cov],
                             dtype=np.int64)
    fully = final_covered >= n_nodes
    n_full = int(fully.sum())
    sat_deg = degrees[fully]  # out-degree = edge count at full coverage

    dec_path = f"{args.out}_saturation_deciles.csv"
    percentiles = list(range(10, 101, 10))
    with open(dec_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["percentile of fully-covered sources",
                    "edges to reach full coverage"])
        if n_full:
            qs = np.percentile(sat_deg, percentiles, method="higher")
            for pct, q in zip(percentiles, qs):
                w.writerow([pct, int(q)])
    print(f"Wrote {dec_path}")

    # Console summary: how tightly peaked the saturation is.
    frac_full = n_full / n_sources
    print(f"  Fully covered: {n_full}/{n_sources} sources ({frac_full:.1%})")
    if n_full:
        p10, p50, p90 = np.percentile(sat_deg, [10, 50, 90], method="higher")
        print(f"  Saturation out-degree deciles: p10={int(p10)}  "
              f"p50={int(p50)}  p90={int(p90)}  "
              f"(p90/p10 spread = {p90 / max(p10, 1):.1f}x)")
        print("  -> a small p90/p10 spread means a tight cliff; large means gradual.")


if __name__ == "__main__":
    main()
