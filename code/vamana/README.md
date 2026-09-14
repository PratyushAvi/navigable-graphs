# Vamana (project copy, with gamma stopping rule)

This project's own copy of ParlayANN's Vamana index, so the algorithm can be
modified and version-controlled here rather than as uncommitted edits inside the
`ParlayANN/` checkout.

`index.h` and `neighbors.h` were copied from ParlayANN at commit
`a48c6a34e5f959190992aa531cc18454cfcc3dd3` (`algorithms/vamana/`). ParlayANN is MIT-licensed and
the original copyright headers are kept intact.

## What is and isn't here

Only the Vamana-specific sources. The supporting headers (`algorithms/utils/`,
`algorithms/bench/`, and `parlaylib`) are still used from the ParlayANN checkout
rather than duplicated — they are a large body of upstream code that this project
does not modify.

Because of that, **the ParlayANN checkout is still required to build**, and the
build also depends on the local arm64/macOS portability fixes in that checkout
(`utils/graph.h`, `utils/point_range.h`, `utils/NSGDist.h`,
`bench/parallelDefsANN`), which are currently uncommitted there.

## Build

```sh
cd code/vamana
make                      # expects ../../ParlayANN
make PARLAYANN=/path/to/ParlayANN   # or point it elsewhere
```

`-I .` precedes ParlayANN's include path, so the headers here take precedence
over the originals in `ParlayANN/algorithms/vamana/`.

## Run

Same interface as upstream; this is what `Vamana-Runs.ipynb` shells out to:

```sh
./neighbors -R 32 -L 64 -alpha 1.0 \
    -data_type float -dist_func Euclidian \
    -base_path <base.fbin> -graph_outfile <graph>
```

## Equivalence with ParlayANN

This copy is functionally identical to `ParlayANN/algorithms/vamana/`. The only
edits are include paths (`"../utils/x.h"` -> `"utils/x.h"`), since the Makefile
puts `algorithms/` on the include path instead of relying on the file sitting
inside that directory. Verified four ways:

1. **Source** — every differing line is an `#include`; normalizing those paths
   makes both files hash identically to upstream.
2. **Preprocessed** — `g++ -E -P` on `neighbors.h` (which transitively pulls in
   `index.h` and all of `utils/`) yields byte-identical 3.9M translation units,
   md5 `b4c3d3cfda9701e29363ebaca9932a9e`. This proves the rewritten includes
   resolve to the same headers, not merely that the text looks similar.
3. **Binary** — identical `__TEXT` (4112384) and `__DATA` (49152) section sizes.
4. **Output** — on fashion_mnist (60000 points), graphs are byte-identical:

   | R | L | alpha | dist_func | md5 |
   |---|---|-------|-----------|-----|
   | 32 | 64 | 1.0 | Euclidian | `3ba0cc6187e07a918a0767d96c5d81cc` |
   | 64 | 128 | 1.2 | Euclidian | `04418193566c886b5e9c5d0906873a0a` |
   | 16 | 32 | 1.0 | Euclidian | `9633af003f976d5d057f2f9b7d33fa79` |
   | 32 | 64 | 1.0 | mips | `2a3404c8db23b712cdd614b02ef3c406` |

Re-run any of these after modifying the algorithm to see exactly what changed:

```sh
./neighbors -R 32 -L 64 -alpha 1.0 -data_type float -dist_func Euclidian \
    -base_path <base.fbin> -graph_outfile /tmp/g_new
cmp /tmp/g_new /tmp/g_baseline
```


## The gamma stopping rule

`robustPrune` normally stops adding edges at `-R`. With `-gamma` it instead stops
once the chosen edges alpha-cover a gamma fraction of a uniform random sample of
the dataset.

```sh
./neighbors -R 32 -L 64 -alpha 1.0 -gamma 0.9 \
    -data_type float -dist_func Euclidian \
    -base_path <base.fbin> -graph_outfile <graph>
```

| flag | meaning |
|------|---------|
| `-gamma <g>` | fraction of the sample to cover, in (0, 1]. Omitted or 0 = stock Vamana. |
| `-S <s>` | sample size. Optional; defaults to `S = ceil(10 ln n)`. |

The sample is drawn once per build (partial Fisher-Yates, uniform without
replacement, fixed seed) and shared by every node, so the denominator is the same
everywhere and gamma means the same thing at each node. A sample point `s` counts
as covered by edge `w` when `alpha * d(w, s) <= d(p, s)` — the same alpha-domination
test robustPrune already applies to candidates.

### R is still a cap

`-R` no longer drives the stopping decision, but it does still bound the degree.
`Graph` allocates `n * (R + 1)` slots and `update_neighbors` calls `abort()` past
`maxDeg`, so letting gamma raise the degree without limit would crash. Selection
stops at whichever comes first: gamma reached, candidates exhausted, or R edges.
Raise `-R` if you want gamma to have more room.

### Measured behaviour

fashion_mnist (n = 60000, R = 32, L = 64, alpha = 1.0), default S = 111:

| gamma | mean degree | true full-dataset coverage |
|-------|-------------|----------------------------|
| (off) | 11.46 | — |
| 0.5 | 3.10 | 58.15% |
| 0.8 | 4.35 | 60.20% |
| 0.9 | 5.21 | 61.52% |
| 1.0 | 8.79 | 65.28% |

Degree and true coverage both rise monotonically with gamma. Degree is stable
across sample sizes (S = 50/500/2000 give 5.14/5.17/5.16 at gamma = 0.9), which is
what an unbiased estimator should do.

### Nodes that fall short of gamma

A small fraction of nodes end up below gamma. They are **not** capped by R — they
have low degree (mean 2.75 at gamma = 0.9) because `robustPrune` ran out of
candidates: its candidate set is the beam search result, so a node can exhaust
everything it ever saw while still short of gamma. Widening the beam shrinks this:

| L | nodes short of gamma (of 2000 sampled) |
|---|----------------------------------------|
| 64 | 51 (2.55%) |
| 128 | 25 (1.25%) |
| 256 | 14 (0.70%) |

This is inherent to Vamana's candidate generation, not to the stopping rule.

### Where the code lives

- `index.h` — the sample (`build_sample`) and the stopping rule in `robustPrune`.
- `neighbors.h` — passes `-gamma`/`-S` from `BuildParams` onto the index.
- `patches/gamma-driver.patch` — the two ParlayANN files that must also change:
  `utils/types.h` (two defaulted `BuildParams` fields) and
  `bench/neighborsTime.C` (parsing `-gamma`/`-S`). Both are outside this copy, so
  the patch is kept here; apply with `git apply` from the ParlayANN root. Both
  changes are additive and leave every other algorithm's behaviour untouched.

With `-gamma` absent the binary reproduces the pre-change baseline byte-for-byte
(md5 `3ba0cc6187e07a918a0767d96c5d81cc`), so stock Vamana runs are unaffected.
