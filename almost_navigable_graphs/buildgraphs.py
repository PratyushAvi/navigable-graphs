"""Almost-navigable graph construction (Algorithm 1).

Builds a gamma-navigable graph G = (P, E) that is correct with probability
1 - delta, using only blackbox access to a distance function d : P x P -> R>=0.

The construction proceeds in rounds. In each round the current "unsettled" set
Pi is partitioned into small subsets of size ceil(4/(1-gamma)). A multiset W of
w = 16 log(n/delta) / (1-gamma) random witnesses is drawn from P. Within each
subset S_j, every point v claims the witnesses whose nearest neighbour *in S_j*
is v. If v claims few enough witnesses (<= (1-gamma) w / 2) it is "settled": it
gets an out-edge to every point in its subset. Otherwise it is deferred to the
next round. Once Pi is small enough, its remaining points are connected to all
of P.

Reference: Algorithm 1, "Almost navigable graph construction".
"""

import math
import argparse

import numpy as np
import h5py
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Distance functions
# --------------------------------------------------------------------------- #
def euclidean_pairwise(P):
    """Return a BLAS-optimized *squared*-Euclidean distance oracle for P (n, d).

    The returned callable dmat(rows, cols) gives the (|rows|, |cols|) matrix of
    SQUARED Euclidean distances ||P[r] - P[c]||^2. The algorithm only ever feeds
    these distances into argmin / '<' comparisons, for which squared distance is
    order-equivalent to distance, so the sqrt is skipped.

    The core computation is a single float32 GEMM (the cross term P[rows] @ P[cols].T,
    dispatched to the NumPy BLAS backend / OpenBLAS-MKL), plus O(k) precomputed
    squared norms via the identity  ||a-b||^2 = ||a||^2 - 2 a.b + ||b||^2. Set
    OMP_NUM_THREADS / OPENBLAS_NUM_THREADS to control BLAS parallelism.

    Swap in any callable with the same signature for a different metric.
    """
    P = np.ascontiguousarray(P, dtype=np.float32)
    sq = np.einsum('ij,ij->i', P, P)          # squared norms, (n,) float32

    def dmat(rows, cols):
        A = P[rows]                             # (|rows|, d) gather
        B = P[cols]                             # (|cols|, d) gather
        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2; the a.b term is the BLAS GEMM.
        d2 = A @ B.T                            # (|rows|, |cols|) float32 GEMM
        d2 *= -2.0
        d2 += sq[rows][:, None]
        d2 += sq[cols][None, :]
        np.maximum(d2, 0.0, out=d2)             # guard tiny negatives
        return d2

    return dmat


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #
def build_almost_navigable_graph(n, dmat, gamma, delta, rng=None, progress=True):
    """Construct a gamma-navigable graph via Algorithm 1.

    Args:
        n:      number of points in P (points are identified by index 0..n-1).
        dmat:   distance oracle. dmat(rows, cols) -> (|rows|, |cols|) ndarray of
                distances d(P[r], P[c]); see euclidean_pairwise for an example.
                Only used via argmin / '<', so any order-equivalent quantity
                (e.g. squared distance) is acceptable.
        gamma:  navigability slack, in [0, 1).
        delta:  failure probability, in (0, 1).
        rng:    optional numpy Generator for the random witness draws (line 5).
        progress: show a tqdm progress bar tracking settled points (default True).

    Returns:
        E: set of directed edges (v, u), meaning an out-edge from v to u.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0, 1); got {gamma}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1); got {delta}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    if rng is None:
        rng = np.random.default_rng()

    # Line 1: initialisation.
    E = set()
    Pi = np.arange(n, dtype=np.int64)                       # Pi^(0) = P
    subset_size = math.ceil(4.0 / (1.0 - gamma))            # ceil(4/(1-gamma))
    w = int(math.ceil(16.0 * math.log(n / delta) / (1.0 - gamma)))  # witnesses
    claim_threshold = (1.0 - gamma) * w / 2.0               # (1-gamma) w / 2

    all_points = np.arange(n, dtype=np.int64)

    # Progress tracks how many points have left Pi (settled or connect-to-all),
    # counting up to n. Postfix shows the round index and the remaining |Pi|.
    pbar = tqdm(total=n, desc="Settling points", unit="pt", disable=not progress)
    round_i = 0

    # Line 2: iterate while Pi is still large enough to partition.
    while len(Pi) >= subset_size:
        prev_pi = len(Pi)

        # Line 3: arbitrarily partition Pi into full subsets of size subset_size
        # plus a leftover set S_bar with < subset_size points.
        k = len(Pi) // subset_size
        split = k * subset_size
        subsets = Pi[:split].reshape(k, subset_size)        # S_1 .. S_k
        S_bar = Pi[split:]                                  # leftovers

        # Line 4: leftovers carry forward to the next round.
        next_Pi = [S_bar]

        # Line 5: draw w witnesses uniformly at random from P (with replacement).
        W = rng.integers(0, n, size=w, dtype=np.int64)

        # Lines 6-14: settle or defer each v in each subset.
        for S_j in tqdm(subsets, desc=f"round {round_i} subsets", leave=False,
                        disable=not progress):
            # For every witness p in W, find its nearest point within S_j
            # (line 8 defines U_hat as this Voronoi-in-S_j assignment).
            #   d_ws[p_idx, s_idx] = d(W[p_idx], S_j[s_idx])
            d_ws = dmat(W, S_j)                              # (w, subset_size)
            nearest_in_S = np.argmin(d_ws, axis=1)          # (w,) index into S_j

            # |U_hat_{S_j, v}| = number of witnesses whose nearest S_j point is v.
            claims = np.bincount(nearest_in_S, minlength=len(S_j))  # (subset_size,)

            for local_v, v in enumerate(S_j):
                if claims[local_v] <= claim_threshold:
                    # Line 10: v is settled -> out-edge to every u in S_j.
                    for u in S_j:
                        if u != v:
                            E.add((int(v), int(u)))
                else:
                    # Line 12: defer v to the next round.
                    next_Pi.append(np.array([v], dtype=np.int64))

        Pi = np.concatenate(next_Pi) if next_Pi else np.empty(0, dtype=np.int64)

        # This round settled (prev_pi - len(Pi)) points.
        pbar.update(prev_pi - len(Pi))
        round_i += 1
        pbar.set_postfix(round=round_i, remaining=len(Pi))

    # Line 18: connect any remaining points to all of P.
    for v in Pi:
        for u in all_points:
            if u != v:
                E.add((int(v), int(u)))

    # The remaining Pi points are now handled (connected to all of P).
    pbar.update(len(Pi))
    pbar.set_postfix(round=round_i, remaining=0)
    pbar.close()

    # Line 19: return G = (P, E).
    return E


# --------------------------------------------------------------------------- #
# CLI / demo driver
# --------------------------------------------------------------------------- #
def _adjacency_from_edges(n, E):
    """Convert an edge set into per-vertex out-neighbour lists."""
    adj = [[] for _ in range(n)]
    for v, u in E:
        adj[v].append(u)
    for lst in adj:
        lst.sort()
    return adj


def save_adj_list(n, adj, path):
    """Write the graph as a plain adjacency list, one source per line:

        <source_id> [e_1, e_2, ...]

    where e_1.. are the out-neighbours of source_id. Points with no out-edges
    are written with an empty list. Parsed by beam_search_almost.py.
    """
    with open(path, 'w') as f:
        for source in range(n):
            neighbours = [int(u) for u in adj[source]]
            f.write(f"{source} {neighbours}\n")


def check_gamma_navigable(n, dmat, adj, gamma, eps=1e-9):
    """Empirically verify gamma-navigability of a built graph.

    gamma-navigability: for each source s, greedy routing must make progress
    toward at least a gamma fraction of all targets t != s. Progress toward t
    exists when some out-neighbour u of s is strictly closer to t than s is,
    i.e. d(u, t) < d(s, t). A direct edge (s, t) also counts (distance 0).

    Returns:
        (min_fraction, frac_sources_ok): the smallest per-source progress
        fraction over all sources, and the fraction of sources that individually
        meet the >= gamma requirement (should be 1.0 for a valid graph).
    """
    D = dmat(np.arange(n), np.arange(n))
    per_source = np.empty(n)
    for s in range(n):
        nbrs = adj[s]
        if not nbrs:
            per_source[s] = 0.0
            continue
        # progress[u_idx, t] = neighbour u is strictly closer to t than s is
        progress = D[nbrs] < D[s][None, :] - eps      # (deg, n)
        reachable = np.any(progress, axis=0)          # (n,)
        reachable[s] = True                           # ignore t == s
        per_source[s] = (reachable.sum() - 1) / (n - 1)
    frac_sources_ok = float(np.mean(per_source >= gamma - eps))
    return float(per_source.min()), frac_sources_ok


def main():
    parser = argparse.ArgumentParser(
        description="Build an almost-navigable graph (Algorithm 1) over a "
                    "point set, using Euclidean distance.")
    parser.add_argument('--dataset', type=str, default=None,
                        help="Path to an HDF5 file (loads the 'train' group, "
                             "shape (n, d)), as in simulrun.py.")
    parser.add_argument('--hdf5_group', type=str, default='train',
                        help="HDF5 group to load points from (default: 'train').")
    parser.add_argument('--input', type=str, default=None,
                        help="Path to an .npy file of shape (n, d). Ignored if "
                             "--dataset is given.")
    parser.add_argument('--n', type=int, default=1000,
                        help="Points to generate when neither --dataset nor --input is given.")
    parser.add_argument('--dim', type=int, default=16,
                        help="Dimensionality when generating random points.")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="Navigability slack gamma in [0, 1).")
    parser.add_argument('--delta', type=float, default=0.01,
                        help="Failure probability delta in (0, 1).")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed for witness draws / point generation.")
    parser.add_argument('--output', type=str, default=None,
                        help="Optional path to save the adjacency list "
                             "('<source> [e_1, e_2, ...]' per line).")
    parser.add_argument('--verify', action='store_true',
                        help="Empirically check gamma-navigability of the result "
                             "(O(n^2) memory; use on modest n).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.dataset is not None:
        with h5py.File(args.dataset, 'r') as f:
            P = f[args.hdf5_group][:]
        print(f"Loaded '{args.hdf5_group}' from {args.dataset}: shape {P.shape}")
    elif args.input is not None:
        P = np.load(args.input)
    else:
        P = rng.standard_normal((args.n, args.dim))
    n = P.shape[0]
    print(f"Building almost-navigable graph over n={n} points "
          f"(gamma={args.gamma}, delta={args.delta})")

    dmat = euclidean_pairwise(P)
    E = build_almost_navigable_graph(n, dmat, args.gamma, args.delta, rng=rng)

    adj = _adjacency_from_edges(n, E)
    out_degrees = np.array([len(a) for a in adj])
    print(f"Edges: {len(E)}  |  out-degree min/mean/max = "
          f"{out_degrees.min()}/{out_degrees.mean():.1f}/{out_degrees.max()}")

    if args.verify:
        min_frac, frac_ok = check_gamma_navigable(n, dmat, adj, args.gamma)
        status = "OK" if frac_ok >= 1.0 - 1e-9 else "FAILED"
        print(f"Navigability [{status}]: {100*frac_ok:.1f}% of sources meet "
              f">= gamma={args.gamma}; worst per-source progress fraction = {min_frac:.3f}")

    if args.output is not None:
        save_adj_list(n, adj, args.output)
        print(f"Saved adjacency list -> {args.output}")


if __name__ == '__main__':
    main()
