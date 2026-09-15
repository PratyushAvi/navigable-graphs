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

# --------------------------------------------------------------------------
# CONFIG -- edit these, or override any of them with the matching CLI flag.
# --------------------------------------------------------------------------
CONFIG = {
    # paths
    "base_fbin":   "/scratch/pa2439/ANN-Search/datasets/glove25-25-angular/base.fbin",
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
    "sample_size": None,        # -S; None = binary default, ceil(10 ln n)

    # coverage / adj-list
    "coverage_alpha": 1.0,
    "limit": None,              # cap nodes, for a quick check
    "dtype": "float64",
    "chunk": 100,

    # stats sweeps
    "cov_min": 90.0, "cov_max": 100.0, "cov_step": 0.5,   # coverage -> degree
    "edge_min": 1,   "edge_max": None, "edge_step": 1,     # edges -> coverage

    # beam search (off unless --beam-widths is given)
    "beam_widths": [],
    "beam_queries": 1000,
    "beam_seed": 0,
    "recall_ks": [1, 10, 100],
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
    that is the bulk of the setup cost, and it is identical across gammas. The
    per-source distance row d(source, ยท) is also identical across graphs, so it
    is cached and reused rather than recomputed once per gamma.
    """

    def __init__(self, base_fbin, dtype="float64", alpha=1.0, chunk=100,
                 cache_rows=True):
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
        """d(source, ยท)^2 for every point. Same for every graph in the sweep."""
        hit = self._row_cache.get(source)
        if hit is not None:
            return hit
        p = self.V[source]
        row = self.norms - 2.0 * (self.V @ p) + p @ p
        if self.cache_rows:
            self._row_cache[source] = row
        return row

    def neighborhood_with_uncov(self, source, neighbors):
        xp = self.xp
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
                   report_every=10000):
    """Write the (neighbor, uncov) adj-list. Resumable. Returns wall seconds."""
    indptr, neighbors, _ = read_csr(graph_path)
    total = len(indptr) - 1
    if limit:
        total = min(total, limit)

    start_at = completed_nodes(out_path) if resume else 0
    if start_at >= total:
        print(f"  adj-list complete ({start_at:,} nodes), skipping", flush=True)
        return 0.0
    if start_at:
        print(f"  resuming adj-list at node {start_at:,}/{total:,}", flush=True)

    t0 = time.perf_counter()
    mode = "a" if start_at else "w"
    with open(out_path, mode) as out:
        for i in range(start_at, total):
            nbrs = neighbors[indptr[i]:indptr[i + 1]]
            rec = engine.neighborhood_with_uncov(i, nbrs)
            out.write(f"{i} {rec}\n")
            out.flush()
            if (i + 1) % report_every == 0:
                el = time.perf_counter() - t0
                done = i + 1 - start_at
                rate = done / el if el else 0
                eta = (total - i - 1) / rate if rate else 0
                print(f"    {i+1:,}/{total:,}  {rate:.0f} nodes/s  eta {eta/60:.1f}m",
                      flush=True)
    return time.perf_counter() - t0


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------
def iter_adj(path):
    """Yield (source, [(neighbor, uncov), ...]) from an adj-list file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sp = line.index(" ")
            yield int(line[:sp]), ast.literal_eval(line[sp + 1:])


def coverage_to_degree_rows(path, n_nodes, cov_levels):
    """coverage_to_degree_analysis.py's view: degree stats at each coverage level."""
    n_cov = len(cov_levels)
    thresholds = [(1.0 - c / 100.0) * n_nodes + 1e-6 for c in cov_levels]
    out_deg, in_deg, sources = [], collections.defaultdict(
        lambda: np.zeros(n_cov, dtype=np.int64)), 0

    for _src, nbrs in iter_adj(path):
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


def edge_to_coverage_rows(path, n_nodes, edge_levels):
    """edge_to_coverage_analysis.py's view: coverage reached at each edge count."""
    n_lv = len(edge_levels)
    cov_at = [[] for _ in range(n_lv)]
    sources = 0
    for _src, nbrs in iter_adj(path):
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


def summary_stats(path, n_nodes):
    """Per-graph summary: coverage extremes, degree, total edges."""
    degs, covs, tot = [], [], 0
    for _src, nbrs in iter_adj(path):
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
def beam_search_rows(graph_path, engine, query_fbin, beam_widths, recall_ks,
                     n_queries, seed):
    """Recall / seen / expanded per beam width, averaged over queries.

    Uses the graph file's CSR directly, so it does not depend on the adj-list.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "beam_search"))
    from beam_search import classicBeamSearch           # noqa: E402

    indptr, neighbors, _ = read_csr(graph_path)
    Q = read_fbin(query_fbin)
    rng = np.random.default_rng(seed)
    if n_queries < len(Q):
        Q = Q[rng.choice(len(Q), n_queries, replace=False)]

    xp = engine.xp
    V, norms = engine.V, engine.norms
    Vh = np.asarray(V.get() if engine.on_gpu else V)
    acc = {(bw, k): {"relevant": 0, "seen": 0, "expanded": 0, "q": 0}
           for bw in beam_widths for k in recall_ks}
    maxk = max(recall_ks)

    t0 = time.perf_counter()
    for qi in range(len(Q)):
        q = np.asarray(Q[qi], dtype=Vh.dtype)
        d_q = (np.einsum("ij,ij->i", Vh, Vh) - 2.0 * (Vh @ q) + q @ q
               if qi == 0 else None)
        if d_q is None:
            d_q = np.einsum("ij,ij->i", Vh, Vh) - 2.0 * (Vh @ q) + q @ q
        true_top = np.argsort(d_q)[:maxk]
        for bw in beam_widths:
            k_ret = min(bw, maxk)
            res, expanded, seen = classicBeamSearch(
                0, -1, (indptr, neighbors), d_q, bw, k_ret)
            ret = np.array([node for _, node in sorted(res, key=lambda x: -x[0])])
            for k in recall_ks:
                a = acc[(bw, k)]
                a["relevant"] += len(np.intersect1d(ret[:k], true_top[:k]))
                a["seen"] += seen
                a["expanded"] += expanded
                a["q"] += 1
    wall = time.perf_counter() - t0

    rows = []
    for (bw, k), a in acc.items():
        if not a["q"]:
            continue
        rows.append({
            "beam_width": bw, "k": k,
            "recall": round(a["relevant"] / (a["q"] * k), 6),
            "mean seen": round(a["seen"] / a["q"], 2),
            "mean expanded": round(a["expanded"] / a["q"], 2),
            "queries": a["q"] // 1,
            "search wall (s)": round(wall, 3),
        })
    return rows


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
STAT_COLUMNS = [
    "dataset", "metric", "method", "gamma", "alpha", "R", "L", "dimensions",
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
    "dataset", "metric", "method", "gamma", "alpha", "R", "L",
    "beam_width", "k", "recall", "mean seen", "mean expanded", "queries",
    "search wall (s)",
]


@dataclass
class RunSpec:
    method: str
    gamma: float | None
    graph: Path
    adj: Path
    build_s: float = 0.0
    wall_s: float = 0.0
    adj_s: float = 0.0


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
    p.add_argument("--beam-widths", type=int, nargs="+",
                   help="run beam search at these widths (default: skip search)")
    p.add_argument("--beam-queries", type=int); p.add_argument("--beam-seed", type=int)
    p.add_argument("--skip-baseline", action="store_true",
                   help="sweep gamma only, no stock Vamana run")
    p.add_argument("--no-adjlist", action="store_true",
                   help="build graphs only; skip coverage and stats")
    p.add_argument("--no-row-cache", action="store_true",
                   help="do not cache per-source distance rows (lower memory)")
    p.add_argument("--rebuild", action="store_true",
                   help="rebuild graphs even when the file already exists")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

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
    for g in frange(cfg["gamma_min"], cfg["gamma_max"], cfg["gamma_step"]):
        runs.append(RunSpec("mod-vamana", g,
                            out_dir / f"graph-modvamana-{tagbase}-gamma{g:g}",
                            out_dir / f"adj-list-modvamana-{tagbase}-gamma{g:g}.txt"))

    print(f"\n{len(runs)} runs planned:")
    for r in runs:
        print(f"  {r.method:12s} gamma={r.gamma if r.gamma is not None else '-':<6} "
              f"-> {r.graph.name}")

    # ---- build -----------------------------------------------------------
    print("\n=== building graphs ===")
    for r in runs:
        label = f"{r.method} gamma={r.gamma}" if r.gamma is not None else r.method
        if r.graph.exists() and not args.rebuild:
            print(f"[{label}] graph exists, skipping build "
                  f"(--rebuild to force)", flush=True)
            continue
        print(f"[{label}]", flush=True)
        binary = cfg["mod_vamana_bin"] if r.gamma is not None else cfg["vamana_bin"]
        r.build_s, r.wall_s = run_vamana(
            binary, cfg["base_fbin"], r.graph, cfg["R"], cfg["L"], cfg["alpha"],
            gamma=r.gamma, sample_size=cfg["sample_size"], verbose=args.verbose)
        print(f"  build {r.build_s:.3f}s (wall {r.wall_s:.3f}s)", flush=True)

    if args.no_adjlist:
        print("\n--no-adjlist given; stopping after builds.")
        return

    # ---- coverage: one engine shared by every run ------------------------
    print("\n=== adj-lists ===")
    engine = CoverageEngine(cfg["base_fbin"], dtype=cfg["dtype"],
                            alpha=cfg["coverage_alpha"], chunk=cfg["chunk"],
                            cache_rows=not args.no_row_cache)
    n_nodes = engine.n
    for r in runs:
        label = f"{r.method} gamma={r.gamma}" if r.gamma is not None else r.method
        print(f"[{label}] -> {r.adj.name}", flush=True)
        r.adj_s = write_adj_list(engine, r.graph, r.adj, limit=cfg["limit"])
        if r.adj_s:
            print(f"  adj-list {r.adj_s:.1f}s", flush=True)

    # ---- stats -----------------------------------------------------------
    print("\n=== stats ===")
    cov_levels = frange(cfg["cov_min"], cfg["cov_max"], cfg["cov_step"])
    stats_path = out_dir / f"sweep-stats-{tagbase}.csv"
    search_path = out_dir / f"sweep-search-{tagbase}.csv"
    dims = int(engine.V.shape[1])

    with open(stats_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STAT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            label = f"{r.method} gamma={r.gamma}" if r.gamma is not None else r.method
            summ = summary_stats(r.adj, n_nodes)
            if not summ:
                print(f"[{label}] empty adj-list, skipped", flush=True)
                continue
            emax = cfg["edge_max"] or summ["graph max out degree"]
            edge_levels = list(range(cfg["edge_min"], emax + 1, cfg["edge_step"]))

            c_rows, sources = coverage_to_degree_rows(r.adj, n_nodes, cov_levels)
            e_rows, _ = edge_to_coverage_rows(r.adj, n_nodes, edge_levels)
            base = {
                "dataset": cfg["dataset"], "metric": cfg["metric"],
                "method": r.method, "gamma": "" if r.gamma is None else r.gamma,
                "alpha": cfg["alpha"], "R": cfg["R"], "L": cfg["L"],
                "dimensions": dims, "sources": sources, "total points": n_nodes,
                "build time (s)": round(r.build_s, 3),
                "build wall (s)": round(r.wall_s, 3),
                "adjlist wall (s)": round(r.adj_s, 3),
                **summ,
            }
            for row in c_rows + e_rows:
                w.writerow({**base, **row})
            print(f"[{label}] {len(c_rows)} coverage rows, {len(e_rows)} edge rows",
                  flush=True)
    print(f"wrote {stats_path}")

    # ---- beam search -----------------------------------------------------
    bws = cfg["beam_widths"]
    if bws:
        print("\n=== beam search ===")
        with open(search_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SEARCH_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in runs:
                label = (f"{r.method} gamma={r.gamma}" if r.gamma is not None
                         else r.method)
                print(f"[{label}] widths {bws}", flush=True)
                rows = beam_search_rows(
                    r.graph, engine, cfg["query_fbin"], bws, cfg["recall_ks"],
                    cfg["beam_queries"], cfg["beam_seed"])
                base = {"dataset": cfg["dataset"], "metric": cfg["metric"],
                        "method": r.method,
                        "gamma": "" if r.gamma is None else r.gamma,
                        "alpha": cfg["alpha"], "R": cfg["R"], "L": cfg["L"]}
                for row in rows:
                    w.writerow({**base, **row})
                for row in rows:
                    print(f"    bw={row['beam_width']:<4} k={row['k']:<4} "
                          f"recall={row['recall']:.4f} seen={row['mean seen']:.0f}",
                          flush=True)
        print(f"wrote {search_path}")

    print("\ndone.")


if __name__ == "__main__":
    main()
