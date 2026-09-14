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

## Baseline

Verified against upstream before any modification: on fashion_mnist
(60000 points, R=32, L=64, alpha=1.0) both binaries produce byte-identical graph
files (md5 `3ba0cc6187e07a918a0767d96c5d81cc`), average degree 11.46, max 32.
Rerun that comparison after changing the algorithm to see exactly what your change
altered.
