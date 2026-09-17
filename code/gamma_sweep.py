#!/usr/bin/env python3
"""Sweep gamma for modified Vamana, alongside a stock Vamana baseline.

For each gamma it builds an independent graph -- a gamma_1 < gamma_2 graph is NOT
a prefix of the gamma_2 graph, because gamma changes every node's degree, which
changes the partially built graph that later beam searches traverse, and because
back-edges are appended without passing through robustPrune. So each gamma needs
its own build.

Per graph it writes the (neighbor, uncov) adj-list, then a stats CSV indexed by
(dataset, method, gamma). Optionally runs beam search for search quality.

Everything you are likely to change lives in CONFIG below, or can be overridden
on the command line.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:                      # progress bars are a convenience
    def tqdm(it=None, **kw):
        return it if it is not None else _NullBar()

    class _NullBar:
        def update(self, n=1): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

# --------------------------------------------------------------------------
# CONFIG -- edit these, or override any of them with the matching CLI flag.
# --------------------------------------------------------------------------
CONFIG = {
    # paths
    "base_fbin":   "/scratch/pa2439/ANN-Search/datasets/glove25-25-angular/base.fbin",
    # ann-benchmarks HDF5 for this dataset: it ships `neighbors`/`distances`, so
    # ground truth is converted from it rather than recomputed. Leave None to
    # fall back to --gt-path or ParlayANN's compute_groundtruth.
    "hdf5_path":   "/scratch/pa2439/ANN-Search/datasets/glove25-25-angular.hdf5",
    "gt_path":     None,          # default: <out_dir>/<dataset>.gt
    "gt_k":        100,           # ground-truth depth; ann-benchmarks ships 100
    "groundtruth_bin": "../ParlayANN/data_tools/compute_groundtruth",
    "query_fbin":  "/scratch/pa2439/ANN-Search/datasets/glove25-25-angular/query.fbin",
    "out_dir":     "/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep",
    "vamana_bin":     "../ParlayANN/algorithms/vamana/neighbors",  # stock
    "mod_vamana_bin": "vamana/neighbors",                          # ours, has -gamma
    "dataset":     "glove25-25",
    "metric":      "euclidean",

    # build params
    "R": 32,
    "L": 64,
    "alpha": 1.0,

    # gamma grid (inclusive of max when it lands on the grid)
    "gamma_min":  0.5,
    "gamma_max":  1.0,
    "gamma_step": 0.1,
    "sample_size": None,        # -S; None = binary default, ceil(100 ln n)

    # coverage / adj-list
    "coverage_alpha": 1.0,
    "limit": None,              # cap nodes, for a quick check
    "dtype": "float64",
    "chunk": 100,
    # False skips the exact adj-list pass entirely and estimates the statistics
    # from a sample instead. The exact pass is O(n) distances per edge; the
    # sampled one is O(S), so it runs in minutes rather than hours.
    "adjlist": True,
    "stats_sample": None,       # sample size; None = ceil(100 ln n)

    # stats sweeps
    "cov_min": 90.0, "cov_max": 100.0, "cov_step": 0.5,   # coverage -> degree
    "edge_min": 1,   "edge_max": None, "edge_step": 1,     # edges -> coverage

    # beam search (off unless --beam-widths is given)
    # ParlayANN's harness takes a single -k (its `allr` is a one-element list),
    # so the binary is invoked once per k. It sweeps its own beam-width list
    # (10..1000, filtered to Q >= k) within each. gt_k above must be >= max(k).
    # k=100 aborts on graphs whose reachable set is smaller than 100 -- which
    # includes stock Vamana at R=32 -- so the default stops at 10. Add 100 with
    # --search-k when R is large enough to support it.
    "search_k": [1, 10],
}

BUILD_TIME_RE = re.compile(r"Graph built in ([0-9.]+) seconds")


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_fbin(path):
    with open(path, "rb") as f:
        n, dims = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=np.float32).reshape(n, dims)


def read_csr(path):
    """ParlayANN graph file -> (indptr, neighbors, degrees). Ragged already."""
    with open(path, "rb") as f:
        n, _max_deg = np.fromfile(f, dtype=np.uint32, count=2)
        sizes = np.fromfile(f, dtype=np.uint32, count=int(n)).astype(np.int64)
        edges = np.fromfile(f, dtype=np.uint32)
    indptr = np.empty(len(sizes) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(sizes, out=indptr[1:])
    return indptr, edges.astype(np.int32), sizes


def resolved_sample_size(sample_size, n):
    """The S the binary will actually use.

    -S <= 0 (or omitted) makes build_sample pick ceil(100 ln n), clamped to n.
    Recording the resolved number rather than None keeps a run that relied on
    the default and one that passed the same value explicitly from looking like
    different configurations in the CSV.
    """
    if sample_size and sample_size > 0:
        return min(int(sample_size), int(n))
    return min(int(math.ceil(100.0 * math.log(max(int(n), 2)))), int(n))


def frange(lo, hi, step):
    """Inclusive float grid that tolerates binary-float drift."""
    out, k = [], 0
    while True:
        v = round(lo + k * step, 10)
        if v > hi + 1e-9:
            break
        out.append(round(v, 6))
        k += 1
    return out


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def run_vamana(binary, base_fbin, graph_out, R, L, alpha,
               gamma=None, sample_size=None, verbose=False):
    """Build one graph. Returns (build_seconds, wall_seconds).

    build_seconds is the binary's own timer (excludes reading the fbin and
    writing the graph); wall_seconds is the whole subprocess.
    """
    binary = Path(binary).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found - run `make` in its directory")

    cmd = [str(binary), "-R", str(R), "-L", str(L), "-alpha", str(alpha),
           "-data_type", "float", "-dist_func", "Euclidian",
           "-base_path", str(Path(base_fbin).resolve()),
           "-graph_outfile", str(Path(graph_out).resolve())]
    if gamma is not None:
        cmd += ["-gamma", str(gamma)]
        if sample_size:
            cmd += ["-S", str(sample_size)]
    if verbose:
        cmd += ["-verbose"]

    print("  $", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    build_s = None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        m = BUILD_TIME_RE.search(line)
        if m:
            build_s = float(m.group(1))
        if verbose or "gamma stopping" in line or "average degree" in line:
            print("   ", line.rstrip(), flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"{binary.name} exited with status {proc.returncode}")
    wall = time.perf_counter() - t0
    return (build_s if build_s is not None else wall), wall


# --------------------------------------------------------------------------
# adj-list with uncov
# --------------------------------------------------------------------------
class CoverageEngine:
    """Replays neighbourhoods to get per-edge uncovered counts.

    Vectors and norms are loaded once and reused for every graph in the sweep --
    that is the bulk of the setup cost, and it is identical across gammas.

    `cache_rows` keeps each computed d(source, .) row. It is off by default and
    should stay off: write_adj_list visits every source exactly once per graph,
    so nothing is ever read back, and the cache simply grows by n floats per
    node -- 0.48 MB each on a 60k dataset, so 29 GB by the end of one pass. The
    resulting allocator pressure slows the run down progressively.
    """

    def __init__(self, base_fbin, dtype="float64", alpha=1.0, chunk=100,
                 cache_rows=False):
        self.xp, self.on_gpu = self._pick_backend()
        xp = self.xp
        print(f"loading {base_fbin} ...", flush=True)
        self.V = xp.asarray(read_fbin(base_fbin), dtype=dtype)
        self.norms = xp.einsum("ij,ij->i", self.V, self.V)
        self.n = int(self.V.shape[0])
        self.alpha_sq = float(alpha) ** 2
        self.chunk = chunk
        self.cache_rows = cache_rows
        self._row_cache = {}
        print(f"  {self.n:,} points, dim {self.V.shape[1]}, "
              f"backend {'GPU' if self.on_gpu else 'CPU'}", flush=True)

    @staticmethod
    def _pick_backend():
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            return cp, True
        except Exception:
            return np, False

    def d_source(self, source):
        """d(source, .)^2 for every point."""
        hit = self._row_cache.get(source)
        if hit is not None:
            return hit
        p = self.V[source]
        row = self.norms - 2.0 * (self.V @ p) + p @ p
        if self.cache_rows:
            self._row_cache[source] = row
        return row

    def neighborhood_with_uncov(self, source, neighbors, d_src=None):
        """Replay one node's edges, recording the uncovered count after each.

        `d_src` is d(source, .)^2. It depends only on the source, not on the
        graph, so when several graphs are processed together it is computed once
        and passed in rather than recomputed per graph.
        """
        xp = self.xp
        if d_src is None:
            d_src = self.d_source(source)
        uncov = xp.arange(self.n, dtype=xp.int32)
        uncov = uncov[uncov != source]
        d_uncov = d_src[uncov]

        neighbors = xp.asarray(np.asarray(neighbors, dtype=np.int64))
        out = []
        for start in range(0, len(neighbors), self.chunk):
            block = neighbors[start:start + self.chunk]
            if len(uncov) == 0:
                out.extend((int(w), 0) for w in block.tolist())
                continue
            D = (self.norms[uncov][:, None]
                 - 2.0 * (self.V[uncov] @ self.V[block].T)
                 + self.norms[block][None, :])
            for t, w in enumerate(block.tolist()):
                keep = d_uncov <= D[:, t] * self.alpha_sq
                uncov, d_uncov, D = uncov[keep], d_uncov[keep], D[keep]
                m = uncov != w
                uncov, d_uncov, D = uncov[m], d_uncov[m], D[m]
                out.append((int(w), int(len(uncov))))
        return out


def completed_nodes(path):
    """Leading nodes already written, repairing a tail left half-written."""
    if not os.path.exists(path):
        return 0
    good, done = 0, 0
    with open(path, "r+") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or not line.endswith("\n"):
                break
            try:
                idx, payload = stripped.split(" ", 1)
                if int(idx) != done:
                    break
                ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                break
            good += len(line.encode())
            done += 1
        f.truncate(good)
    return done


def write_adj_list(engine, graph_path, out_path, limit=None, resume=True,
                   desc=None):
    """Write the (neighbor, uncov) adj-list. Resumable. Returns wall seconds."""
    indptr, neighbors, _ = read_csr(graph_path)
    total = len(indptr) - 1
    if limit:
        total = min(total, limit)

    start_at = completed_nodes(out_path) if resume else 0
    if start_at >= total:
        print(f"  adj-list complete ({start_at:,} nodes), skipping", flush=True)
        return None          # measured nothing; do not overwrite an earlier timing
    if start_at:
        print(f"  resuming adj-list at node {start_at:,}/{total:,}", flush=True)

    t0 = time.perf_counter()
    mode = "a" if start_at else "w"
    # initial= makes a resumed run show true overall progress rather than
    # restarting the bar at zero.
    bar = tqdm(total=total, initial=start_at, unit="node", desc=desc or "coverage",
               smoothing=0.05, dynamic_ncols=True,
               mininterval=BAR_INTERVAL, miniters=0, file=BAR_FILE,
               ascii=not _IS_TTY)          # plain characters in a log file
    try:
        with open(out_path, mode) as out:
            for i in range(start_at, total):
                nbrs = neighbors[indptr[i]:indptr[i + 1]]
                rec = engine.neighborhood_with_uncov(i, nbrs)
                out.write(f"{i} {rec}\n")
                out.flush()
                bar.update(1)
    finally:
        bar.close()
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------
def coverage_sample(n, sample_size=None, seed=12345):
    """S point ids drawn uniformly without replacement, default ceil(100 ln n).

    The same rule build_sample uses in index.h, so a graph built with -gamma and
    the statistics measured here are judged against the same size of sample.
    """
    if not sample_size or sample_size <= 0:
        sample_size = int(math.ceil(100.0 * math.log(max(int(n), 2))))
    sample_size = min(int(sample_size), int(n))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(int(n), size=sample_size, replace=False))


def iter_graph_sampled(engine, graph_path, sample, desc=None, limit=None):
    """Yield (source, [(neighbor, uncov_estimate), ...]) straight from a graph.

    `uncov` is the count still uncovered among the sampled points, scaled to the
    dataset, rather than an exact count over all n. One source costs O(S * deg)
    distances instead of O(n * deg), which is what makes this cheap enough to run
    without the adj-list stage.

    Coverage of s by waypoint w uses the same alpha-domination test as the exact
    path: s stays uncovered while d(p, s) <= alpha^2 * d(w, s).
    """
    indptr, neighbors, _ = read_csr(graph_path)
    n = engine.n
    total = len(indptr) - 1
    if limit:
        total = min(total, limit)
    xp = engine.xp
    S = xp.asarray(np.asarray(sample, dtype=np.int64))
    n_s = len(sample)
    scale = n / n_s                       # sampled count -> dataset scale
    VS = engine.V[S]                      # (S, dim), reused for every source
    normsS = engine.norms[S]

    rng = range(total)
    if desc:
        rng = tqdm(rng, desc=desc, unit="node", total=total, smoothing=0.05,
                   dynamic_ncols=True, mininterval=BAR_INTERVAL, miniters=0,
                   file=BAR_FILE, ascii=not _IS_TTY)
    for i in rng:
        nbrs = neighbors[indptr[i]:indptr[i + 1]]
        p = engine.V[i]
        d_ps = normsS - 2.0 * (VS @ p) + p @ p        # d(i, s) for s in sample
        alive = xp.ones(n_s, dtype=bool)
        alive &= (S != i)                             # i covers itself
        out = []
        for w in np.asarray(nbrs, dtype=np.int64).tolist():
            if alive.any():
                vw = engine.V[w]
                d_ws = normsS - 2.0 * (VS @ vw) + vw @ vw
                alive &= (d_ps <= d_ws * engine.alpha_sq)
                alive &= (S != w)
            out.append((int(w), int(round(float(alive.sum()) * scale))))
        yield int(i), out


def iter_adj(path, desc=None, total=None):
    """Yield (source, [(neighbor, uncov), ...]) from an adj-list file.

    Parsing a large adj-list takes a while and each stats pass re-reads it, so
    `desc` puts a progress bar on the scan.
    """
    with open(path) as f:
        if desc and _IS_TTY:
            f = tqdm(f, desc=desc, total=total, unit="node", smoothing=0.05,
                     dynamic_ncols=True, mininterval=BAR_INTERVAL,
                     file=BAR_FILE, leave=False)
        for line in f:
            line = line.strip()
            if not line:
                continue
            sp = line.index(" ")
            yield int(line[:sp]), ast.literal_eval(line[sp + 1:])


def coverage_to_degree_rows(rows, n_nodes, cov_levels):
    """coverage_to_degree_analysis.py's view: degree stats at each coverage level."""
    n_cov = len(cov_levels)
    thresholds = [(1.0 - c / 100.0) * n_nodes + 1e-6 for c in cov_levels]
    out_deg, in_deg, sources = [], collections.defaultdict(
        lambda: np.zeros(n_cov, dtype=np.int64)), 0

    for _src, nbrs in rows:
        sources += 1
        deg_c = [len(nbrs)] * n_cov
        ptr = 0
        for ei, (nb, uncov) in enumerate(nbrs):
            first = ptr
            while ptr < n_cov and uncov <= thresholds[ptr]:
                deg_c[ptr] = ei + 1
                ptr += 1
            in_deg[nb][first:] += 1
            if ptr == n_cov:
                break
        out_deg.append(deg_c)

    if not out_deg:
        return [], 0
    outa = np.asarray(out_deg, dtype=np.int64)
    ina = (np.asarray(list(in_deg.values()), dtype=np.int64)
           if in_deg else np.empty((0, n_cov), dtype=np.int64))
    rows = []
    for ci, cov in enumerate(cov_levels):
        o = outa[:, ci]; o = o[o > 0]
        i_ = ina[:, ci] if len(ina) else np.array([], dtype=np.int64)
        i_ = i_[i_ > 0]
        rows.append({
            "sweep": "coverage_to_degree", "coverage": cov, "edges": "",
            "mean out degree":   round(float(o.mean()), 3) if len(o) else 0.0,
            "median out degree": float(np.median(o)) if len(o) else 0.0,
            "min out degree":    int(o.min()) if len(o) else 0,
            "max out degree":    int(o.max()) if len(o) else 0,
            "median in degree":  float(np.median(i_)) if len(i_) else 0.0,
            "min in degree":     int(i_.min()) if len(i_) else 0,
            "max in degree":     int(i_.max()) if len(i_) else 0,
            "mean points covered": "", "median points covered": "",
            "min points covered": "", "max points covered": "",
            "sources below 99.5% coverage": "", "sources below 100% coverage": "",
        })
    return rows, sources


def edge_to_coverage_rows(rows, n_nodes, edge_levels):
    """edge_to_coverage_analysis.py's view: coverage reached at each edge count."""
    n_lv = len(edge_levels)
    cov_at = [[] for _ in range(n_lv)]
    sources = 0
    for _src, nbrs in rows:
        sources += 1
        deg = len(nbrs)
        for li, e in enumerate(edge_levels):
            if deg == 0:
                cov_at[li].append(0)
                continue
            idx = min(e, deg) - 1          # uncov after the first e edges
            cov_at[li].append(n_nodes - nbrs[idx][1])
    rows = []
    for li, e in enumerate(edge_levels):
        c = np.asarray(cov_at[li], dtype=np.int64)
        if not len(c):
            continue
        rows.append({
            "sweep": "edge_to_coverage", "coverage": "", "edges": e,
            "mean out degree": "", "median out degree": "",
            "min out degree": "", "max out degree": "",
            "median in degree": "", "min in degree": "", "max in degree": "",
            "mean points covered":   round(float(c.mean()), 3),
            "median points covered": float(np.median(c)),
            "min points covered":    int(c.min()),
            "max points covered":    int(c.max()),
            "sources below 99.5% coverage": int((c < 0.995 * n_nodes).sum()),
            "sources below 100% coverage":  int((c < n_nodes).sum()),
        })
    return rows, sources


def summary_stats(rows, n_nodes):
    """Per-graph summary: coverage extremes, degree, total edges."""
    degs, covs, tot = [], [], 0
    for _src, nbrs in rows:
        degs.append(len(nbrs))
        tot += len(nbrs)
        last = nbrs[-1][1] if nbrs else n_nodes
        covs.append(1.0 - last / n_nodes)
    if not degs:
        return {}
    # NB: these keys are deliberately prefixed "graph ". The per-sweep rows
    # carry "median out degree"/"max out degree" of their own, and an unprefixed
    # key here would be overwritten by them when the two dicts are merged.
    return {
        "nodes": len(degs),
        "graph min coverage": round(100.0 * min(covs), 4),
        "graph max coverage": round(100.0 * max(covs), 4),
        "graph avg out degree": round(float(np.mean(degs)), 4),
        "graph median out degree": float(np.median(degs)),
        "graph max out degree": int(max(degs)),
        "graph total edges": tot,
    }


# --------------------------------------------------------------------------
# beam search
# --------------------------------------------------------------------------
def write_gt_file(path, neighbors, distances):
    """Write ParlayANN's ground-truth format.

    [int32 n][int32 k][n*k int32 ids][n*k float32 dists] -- see groundTruth in
    ParlayANN/algorithms/utils/types.h.
    """
    neighbors = np.ascontiguousarray(neighbors, dtype=np.int32)
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    n, k = neighbors.shape
    with open(path, "wb") as f:
        np.array([n, k], dtype=np.int32).tofile(f)
        neighbors.tofile(f)
        distances.tofile(f)
    return n, k


def groundtruth_from_hdf5(hdf5_path, gt_path, k=None):
    """Convert an ANN-benchmarks HDF5's own ground truth into ParlayANN's format.

    These files ship `neighbors` and `distances` for the `test` queries, so the
    exact top-k is already known and nothing has to be recomputed.

    ParlayANN compares squared euclidean distances internally, while
    ann-benchmarks stores plain euclidean for "-euclidean" datasets, so the
    distances are squared here. Recall only uses the ids, but the distances are
    used for tie handling, so the scale has to match.
    """
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "neighbors" not in f or "distances" not in f:
            raise KeyError(
                f"{hdf5_path} has no 'neighbors'/'distances' datasets; it does "
                f"not carry ground truth. Keys: {list(f.keys())}")
        nbrs = f["neighbors"][:]
        dists = f["distances"][:]
    if k is not None and k < nbrs.shape[1]:
        nbrs, dists = nbrs[:, :k], dists[:, :k]
    n, kk = write_gt_file(gt_path, nbrs, dists.astype(np.float64) ** 2)
    print(f"ground truth from {Path(hdf5_path).name}: {n:,} queries x {kk} -> "
          f"{gt_path}", flush=True)
    return gt_path


def ensure_groundtruth(cfg, gt_path):
    """Resolve ground truth: use --gt-path, else the HDF5's, else compute it."""
    gt_path = Path(gt_path)
    if gt_path.exists():
        print(f"ground truth present: {gt_path}", flush=True)
        return gt_path

    if cfg.get("hdf5_path"):
        return groundtruth_from_hdf5(cfg["hdf5_path"], gt_path, cfg.get("gt_k"))

    gt_bin = Path(cfg["groundtruth_bin"]).resolve()
    if not gt_bin.exists():
        raise FileNotFoundError(
            f"no ground truth available.\n"
            f"  give --hdf5-path (ann-benchmarks files ship their own), or\n"
            f"  give --gt-path pointing at an existing .gt file, or\n"
            f"  build {gt_bin} with: cd ParlayANN/data_tools && "
            f"make compute_groundtruth")
    cmd = [str(gt_bin),
           "-base_path", str(Path(cfg["base_fbin"]).resolve()),
           "-query_path", str(Path(cfg["query_fbin"]).resolve()),
           "-gt_path", str(gt_path.resolve()),
           "-data_type", "float", "-dist_func", "Euclidian",
           "-k", str(cfg.get("gt_k") or 100)]
    print(f"computing ground truth -> {gt_path}", flush=True)
    print("  $", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compute_groundtruth failed:\n{r.stdout}\n{r.stderr}")
    print(f"  done in {time.perf_counter() - t0:.1f}s", flush=True)
    return gt_path


def parse_parlay_res_csv(path):
    """Rows from ParlayANN's result CSV.

    The file is several stacked tables: a graph header, a blank line, a results
    table, then the whole thing repeated -- the harness runs its sweep more than
    once, and the later pass is the warm one (higher QPS). Every results table is
    returned, tagged with `pass_index`, so nothing is silently dropped.

    A row is NOT one beam width. ParlayANN walks its beam list per target recall
    bucket and reports the first width that reached that target, so the table is
    a recall-vs-cost curve: "Target recall" is the index and "Q" is the width it
    needed. Different graphs hit different targets, so row counts vary.
    """
    with open(path, newline="") as fh:
        lines = list(csv.reader(fh))

    rows, pass_i = [], 0
    i = 0
    while i < len(lines):
        row = lines[i]
        if row and row[0] == "Num queries":
            header = row
            i += 1
            while i < len(lines) and lines[i] and lines[i][0].strip():
                rows.append({**dict(zip(header, lines[i])), "pass_index": pass_i})
                i += 1
            pass_i += 1
        else:
            i += 1
    return rows


def parlay_search(binary, graph_path, base_fbin, query_fbin, gt_path, res_path,
                  k, R, L, alpha, verbose=False):
    """Run ParlayANN's own recall harness against an existing graph.

    -graph_path loads the graph instead of building one, so this measures search
    only. ParlayANN sweeps its own built-in beam-width list (10..1000, filtered
    to Q >= k) and reports recall, QPS and the visited/comparison counters.
    """
    binary = Path(binary).resolve()
    cmd = [str(binary), "-R", str(R), "-L", str(L), "-alpha", str(alpha),
           "-data_type", "float", "-dist_func", "Euclidian",
           "-base_path", str(Path(base_fbin).resolve()),
           "-query_path", str(Path(query_fbin).resolve()),
           "-gt_path", str(Path(gt_path).resolve()),
           "-graph_path", str(Path(graph_path).resolve()),
           "-res_path", str(Path(res_path).resolve()),
           "-k", str(k)]
    print("  $", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"search failed:\n{proc.stdout}\n{proc.stderr}")
    if verbose:
        print(proc.stdout, flush=True)
    wall = time.perf_counter() - t0

    rows = parse_parlay_res_csv(res_path)
    if not rows:
        raise RuntimeError(
            f"no result rows in {res_path}. ParlayANN writes the header but no "
            f"rows when ground truth is missing or does not match the dataset.")

    out = []
    for r in rows:
        out.append({
            # ParlayANN's names on the left, this project's on the right.
            # "beam width" is the width that first reached "target recall", not
            # a swept parameter -- see parse_parlay_res_csv.
            "target recall":  float(r.get("Target recall", "nan")),
            "beam width":     int(float(r.get("Q", 0))),
            "k":              int(float(r.get("k", k))),
            "recall":         float(r.get("Actual recall", "nan")),
            "QPS":            float(r.get("QPS", "nan")),
            "mean expanded":  float(r.get("Average Cmps", "nan")),
            "tail expanded":  float(r.get("Tail Cmps", "nan")),
            "mean seen":      float(r.get("Average Visited", "nan")),
            "tail seen":      float(r.get("Tail Visited", "nan")),
            "queries":        int(float(r.get("Num queries", 0))),
            "pass":           r.get("pass_index", 0),
            "search wall (s)": round(wall, 3),
        })
    return out


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
STAT_COLUMNS = [
    "dataset", "metric", "method", "gamma", "alpha", "R", "L", "S", "dimensions",
    "estimated coverage", "coverage sample",
    "sources", "total points", "sweep", "coverage", "edges",
    "mean out degree", "median out degree", "min out degree", "max out degree",
    "median in degree", "min in degree", "max in degree",
    "mean points covered", "median points covered",
    "min points covered", "max points covered",
    "sources below 99.5% coverage", "sources below 100% coverage",
    "build time (s)", "build wall (s)", "adjlist wall (s)",
    "graph min coverage", "graph max coverage", "graph avg out degree",
    "graph median out degree", "graph max out degree", "graph total edges",
]

SEARCH_COLUMNS = [
    "dataset", "metric", "method", "gamma", "alpha", "R", "L", "S",
    "target recall", "beam width", "k", "recall", "QPS",
    "mean seen", "tail seen", "mean expanded", "tail expanded",
    "queries", "pass", "search wall (s)",
]



# Rows from every run accumulate into one CSV per dataset. A rerun of the same
# configuration replaces its own rows rather than appending duplicates, so the
# file can be rebuilt incrementally as gammas are added. Same upsert pattern as
# coverage_to_degree_analysis.py.
STATS_KEY = ["dataset", "metric", "method", "gamma", "alpha", "R", "L", "S",
             "estimated coverage", "sweep", "coverage", "edges"]
SEARCH_KEY = ["dataset", "metric", "method", "gamma", "alpha", "R", "L", "S",
              "k", "target recall", "pass"]


def upsert_csv(path, rows, columns, key_cols):
    """Merge `rows` into the CSV at `path`, replacing rows with matching keys."""
    path = Path(path)
    new = [{c: r.get(c, "") for c in columns} for r in rows]
    if not new:
        return 0, 0

    kept = []
    if path.exists():
        by_key = {}
        for r in new:
            by_key[tuple(str(r.get(c, "")) for c in key_cols)] = r
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                k = tuple(str(r.get(c, "")) for c in key_cols)
                replacement = by_key.get(k)
                if replacement is None:
                    kept.append({c: r.get(c, "") for c in columns})
                    continue
                # This row is being replaced, but a blank field in the new row
                # means "not measured this run" rather than "no value" -- a
                # skipped build records no time, and must not erase the time
                # logged by the run that did the work.
                for c in columns:
                    if replacement.get(c, "") == "" and r.get(c, "") != "":
                        replacement[c] = r[c]

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in kept:
            w.writerow(r)
        for r in new:
            w.writerow(r)
    os.replace(tmp, path)          # atomic: a crash cannot truncate the CSV
    return len(new), len(kept)


_T0 = time.perf_counter()

# tqdm repaints with a carriage return, which only works on a terminal. Under
# slurm the output is a file, so every refresh would be a separate line; refresh
# once a minute there and keep it lively when someone is watching.
_IS_TTY = sys.stdout.isatty()
# Redirected output is block-buffered, and tqdm does not flush, so bar refreshes
# would sit unwritten and the log would look stalled. This forces each one out.
# 0 = refresh on every node. Under slurm that is one line per node rather than a
# repainted bar, which is the point: progress is unambiguous and a stall is
# obvious immediately.
BAR_INTERVAL = 0.0


class _FlushingStream:
    """stdout wrapper that flushes on every write, for tqdm under slurm."""

    def __init__(self, stream):
        self._s = stream

    def write(self, data):
        self._s.write(data)
        self._s.flush()

    def flush(self):
        self._s.flush()

    def __getattr__(self, name):
        return getattr(self._s, name)


BAR_FILE = sys.stdout if _IS_TTY else _FlushingStream(sys.stdout)


def stage(title):
    """Banner marking a stage boundary, stamped with elapsed wall time."""
    el = time.perf_counter() - _T0
    print(f"\n{'=' * 70}\n=== {title}   [+{el/60:.1f} min]\n{'=' * 70}",
          flush=True)


@dataclass
class RunSpec:
    method: str
    gamma: float | None
    graph: Path
    adj: Path
    # None means "not measured in this run" -- the stage was skipped because the
    # work was already done. Writing 0.0 would overwrite the real timing from the
    # run that did it, since the upsert replaces the whole row.
    build_s: float | None = None
    wall_s: float | None = None
    adj_s: float | None = None


def main():
    cfg = dict(CONFIG)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-fbin"); p.add_argument("--query-fbin")
    p.add_argument("--out-dir"); p.add_argument("--dataset"); p.add_argument("--metric")
    p.add_argument("--vamana-bin"); p.add_argument("--mod-vamana-bin")
    p.add_argument("--R", type=int); p.add_argument("--L", type=int)
    p.add_argument("--alpha", type=float)
    p.add_argument("--gamma-min", type=float); p.add_argument("--gamma-max", type=float)
    p.add_argument("--gamma-step", type=float); p.add_argument("--sample-size", type=int)
    p.add_argument("--coverage-alpha", type=float); p.add_argument("--limit", type=int)
    p.add_argument("--dtype"); p.add_argument("--chunk", type=int)
    p.add_argument("--cov-min", type=float); p.add_argument("--cov-max", type=float)
    p.add_argument("--cov-step", type=float)
    p.add_argument("--edge-min", type=int); p.add_argument("--edge-max", type=int)
    p.add_argument("--edge-step", type=int)
    p.add_argument("--search", action="store_true",
                   help="run ParlayANN's recall harness on each graph")
    p.add_argument("--search-k", type=int, nargs="+",
                   help="k values for the recall harness; the binary is run "
                        "once per k (default: 1 10 100)")
    p.add_argument("--hdf5-path",
                   help="ann-benchmarks HDF5; its neighbors/distances become the "
                        "ground truth, so nothing is recomputed")
    p.add_argument("--gt-path", help="existing .gt file to use instead")
    p.add_argument("--gt-k", type=int)
    p.add_argument("--skip-baseline", action="store_true",
                   help="sweep gamma only, no stock Vamana run")
    p.add_argument("--no-adjlist", action="store_true",
                   help="skip the exact adj-list pass; estimate the statistics "
                        "from a sample of the points instead")
    p.add_argument("--stats-sample", type=int,
                   help="points sampled to estimate coverage "
                        "(default: ceil(100 ln n), matching -S in the build)")
    p.add_argument("--row-cache", action="store_true",
                   help="cache per-source distance rows. Off by default: each "
                        "source is visited once per graph, so the cache is "
                        "never read and grows by n floats per node")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild graphs even when the file already exists")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.no_adjlist:
        cfg["adjlist"] = False

    for k, v in vars(args).items():
        if v is not None and k in cfg:
            cfg[k] = v
        elif v is not None and k.replace("_", "") in {c.replace("_", "") for c in cfg}:
            for c in cfg:
                if c.replace("_", "") == k.replace("_", ""):
                    cfg[c] = v

    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    tagbase = f"{cfg['dataset']}-{cfg['metric']}-R{cfg['R']}-alpha{cfg['alpha']:g}"
    print(f"output -> {out_dir}")
    (out_dir / "sweep_config.json").write_text(json.dumps(cfg, indent=2, default=str))

    # ---- plan the runs ---------------------------------------------------
    runs: list[RunSpec] = []
    if not args.skip_baseline:
        runs.append(RunSpec("vamana", None,
                            out_dir / f"graph-vamana-{tagbase}",
                            out_dir / f"adj-list-vamana-{tagbase}.txt"))
    # S is part of the filename: different sample sizes are different graphs, and
    # without it the second one would be skipped as "already built".
    s_tag = "" if not cfg["sample_size"] else f"-S{int(cfg['sample_size'])}"
    for g in frange(cfg["gamma_min"], cfg["gamma_max"], cfg["gamma_step"]):
        runs.append(RunSpec("mod-vamana", g,
                            out_dir / f"graph-modvamana-{tagbase}-gamma{g:g}{s_tag}",
                            out_dir / f"adj-list-modvamana-{tagbase}-gamma{g:g}{s_tag}.txt"))

    print(f"\n{len(runs)} runs planned:")
    for r in runs:
        print(f"  {r.method:12s} gamma={r.gamma if r.gamma is not None else '-':<6} "
              f"-> {r.graph.name}")

    # ---- build -----------------------------------------------------------
    stage("1/4  building graphs")
    for ri, r in enumerate(runs, 1):
        label = f"{r.method} gamma={r.gamma}" if r.gamma is not None else r.method
        label = f"[{ri}/{len(runs)}] {label}"
        if r.graph.exists() and not args.rebuild:
            print(f"{label}: graph exists, skipping build "
                  f"(--rebuild to force)", flush=True)
            continue
        print(f"{label}: building", flush=True)
        binary = cfg["mod_vamana_bin"] if r.gamma is not None else cfg["vamana_bin"]
        r.build_s, r.wall_s = run_vamana(
            binary, cfg["base_fbin"], r.graph, cfg["R"], cfg["L"], cfg["alpha"],
            gamma=r.gamma, sample_size=cfg["sample_size"], verbose=args.verbose)
        print(f"  build {r.build_s:.3f}s (wall {r.wall_s:.3f}s)", flush=True)

    # ---- coverage: one engine shared by every run ------------------------
    # Both paths need it: the exact pass replays against all n points, the
    # sampled one against S of them, and both use its V and norms.
    engine = CoverageEngine(cfg["base_fbin"], dtype=cfg["dtype"],
                            alpha=cfg["coverage_alpha"], chunk=cfg["chunk"],
                            cache_rows=args.row_cache)
    n_nodes = engine.n

    if cfg["adjlist"]:
        stage("2/4  coverage adj-lists")
        # One graph at a time: each adj-list is finished before the next starts,
        # so an interrupted run leaves completed files rather than every file
        # partial, and the per-graph timing is real rather than an average.
        for ri, r in enumerate(runs, 1):
            lbl = (f"{r.method} gamma={r.gamma}" if r.gamma is not None
                   else r.method)
            label = f"[{ri}/{len(runs)}] {lbl}"
            print(f"{label} -> {r.adj.name}", flush=True)
            r.adj_s = write_adj_list(engine, r.graph, r.adj,
                                     limit=cfg["limit"], desc=label)
            if r.adj_s is not None:
                print(f"  adj-list {r.adj_s:.1f}s", flush=True)
    else:
        stage("2/4  coverage adj-lists (skipped)")
        print("  --no-adjlist: statistics are estimated from a sample instead, "
              "straight from the graph files", flush=True)

    # ---- stats -----------------------------------------------------------
    stage("3/4  stats")
    cov_levels = frange(cfg["cov_min"], cfg["cov_max"], cfg["cov_step"])
    sample = (None if cfg["adjlist"]
              else coverage_sample(n_nodes, cfg["stats_sample"]))
    if sample is not None:
        print(f"coverage sample: {len(sample):,} points "
              f"(ceil(100 ln {n_nodes:,}) unless --stats-sample given)",
              flush=True)
    # One accumulating file per dataset: R, alpha, method and gamma are columns,
    # so runs with different build parameters share it instead of each writing a
    # separate CSV.
    stats_path = out_dir / f"sweep-stats-{cfg['dataset']}.csv"
    search_path = out_dir / f"sweep-search-{cfg['dataset']}.csv"
    dims = int(engine.V.shape[1])

    stats_rows = []
    for ri, r in enumerate(runs, 1):
        label = f"{r.method} gamma={r.gamma}" if r.gamma is not None else r.method
        label = f"[{ri}/{len(runs)}] {label}"
        if cfg["adjlist"]:
            # Exact: uncov counted over every point, read back from the adj-list.
            print(f"{label}: scanning {r.adj.name}", flush=True)
            rows = list(iter_adj(r.adj, f"{label}: reading", n_nodes))
        else:
            # Estimated: uncov counted over the sample and scaled to n. Each
            # function below consumes the rows once, so they are materialised.
            print(f"{label}: estimating coverage from {len(sample):,} "
                  f"sampled points", flush=True)
            rows = list(iter_graph_sampled(engine, r.graph, sample,
                                           desc=f"{label}: sampling",
                                           limit=cfg["limit"]))
        summ = summary_stats(rows, n_nodes)
        if not summ:
            print(f"{label}: empty adj-list, skipped", flush=True)
            continue
        emax = cfg["edge_max"] or summ["graph max out degree"]
        edge_levels = list(range(cfg["edge_min"], emax + 1, cfg["edge_step"]))

        c_rows, sources = coverage_to_degree_rows(rows, n_nodes, cov_levels)
        e_rows, _ = edge_to_coverage_rows(rows, n_nodes, edge_levels)
        base = {
            "dataset": cfg["dataset"], "metric": cfg["metric"],
            "method": r.method, "gamma": "" if r.gamma is None else r.gamma,
            "alpha": cfg["alpha"], "R": cfg["R"], "L": cfg["L"],
            # Stock Vamana draws no sample, so S is blank rather than a number
            # that would suggest it influenced the build.
            "S": "" if r.gamma is None else resolved_sample_size(
                cfg["sample_size"], n_nodes),
            "dimensions": dims, "sources": sources, "total points": n_nodes,
            # The coverage columns are exact only when they came from the
            # adj-list pass; otherwise they are scaled from a sample.
            "estimated coverage": 0 if cfg["adjlist"] else 1,
            "coverage sample": "" if cfg["adjlist"] else len(sample),
            "build time (s)": "" if r.build_s is None else round(r.build_s, 3),
            "build wall (s)": "" if r.wall_s is None else round(r.wall_s, 3),
            "adjlist wall (s)": "" if r.adj_s is None else round(r.adj_s, 3),
            **summ,
        }
        for row in c_rows + e_rows:
            stats_rows.append({**base, **row})
        print(f"{label}: {len(c_rows)} coverage rows, {len(e_rows)} edge rows",
              flush=True)

    n_new, n_kept = upsert_csv(stats_path, stats_rows, STAT_COLUMNS, STATS_KEY)
    print(f"wrote {stats_path}: {n_new} rows from this run, "
          f"{n_kept} kept from earlier runs")

    # ---- search (ParlayANN's own recall harness) -------------------------
    if args.search:
        stage("4/4  search")
        search_ks = cfg["search_k"]
        if isinstance(search_ks, int):
            search_ks = [search_ks]
        search_ks = sorted(set(int(k) for k in search_ks))
        # Recall at k needs at least k ground-truth neighbours per query.
        if cfg.get("gt_k") and max(search_ks) > cfg["gt_k"]:
            raise ValueError(
                f"--search-k up to {max(search_ks)} needs gt_k >= that, "
                f"but gt_k is {cfg['gt_k']}")
        print(f"k values: {search_ks}")

        gt_path = cfg.get("gt_path") or (out_dir / f"{cfg['dataset']}.gt")
        gt_path = ensure_groundtruth(cfg, gt_path)
        search_rows = []
        for ri, r in enumerate(runs, 1):
            label = (f"{r.method} gamma={r.gamma}" if r.gamma is not None
                     else r.method)
            label = f"[{ri}/{len(runs)}] {label}"
            print(f"{label}", flush=True)
            base = {"dataset": cfg["dataset"], "metric": cfg["metric"],
                    "method": r.method,
                    "gamma": "" if r.gamma is None else r.gamma,
                    "alpha": cfg["alpha"], "R": cfg["R"], "L": cfg["L"],
                    "S": "" if r.gamma is None else resolved_sample_size(
                        cfg["sample_size"], n_nodes)}
            for k in search_ks:
                # One invocation per k: ParlayANN's harness runs a single -k.
                # The stock binary is fine for searching any graph -- the search
                # code is identical, and -graph_path skips the build entirely.
                # No error handling here on purpose: ParlayANN aborts when beam
                # search cannot return k results, and that abort propagates. A
                # graph too sparse to serve k is a real result, not something to
                # paper over -- use a k the graph can support (see --search-k).
                rows = parlay_search(
                    cfg["vamana_bin"], r.graph, cfg["base_fbin"],
                    cfg["query_fbin"], gt_path,
                    out_dir / f"res-{r.graph.name}-k{k}.csv",
                    k, cfg["R"], cfg["L"], cfg["alpha"], verbose=args.verbose)
                for row in rows:
                    search_rows.append({**base, **row})
                npass = len({r_["pass"] for r_ in rows})
                best = max(rows, key=lambda x: x["recall"])
                print(f"    k={k:<4} {len(rows)} rows over {npass} pass(es); "
                      f"best recall {best['recall']:.4f} needed Q="
                      f"{best['beam width']} "
                      f"(QPS {best['QPS']:.0f}, seen {best['mean seen']:.0f})",
                      flush=True)

        n_new, n_kept = upsert_csv(search_path, search_rows, SEARCH_COLUMNS,
                                   SEARCH_KEY)
        print(f"wrote {search_path}: {n_new} rows from this run, "
              f"{n_kept} kept from earlier runs")

    stage(f"done in {(time.perf_counter() - _T0)/60:.1f} min")


if __name__ == "__main__":
    main()
