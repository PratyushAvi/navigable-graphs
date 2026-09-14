# Vamana (project copy)

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
