#!/bin/bash
# Gamma sweep: stock Vamana + modified Vamana over a gamma grid, then coverage
# adj-lists, stats, and ParlayANN's recall harness.
#
# Submit:            sbatch gamma_sweep.slurm
# Override dataset:  sbatch --export=ALL,DATASET=mnist-784-euclidean gamma_sweep.slurm
# Override grid:     sbatch --export=ALL,GAMMA_MIN=0.8,GAMMA_MAX=1.0,GAMMA_STEP=0.05 gamma_sweep.slurm
#
# The sweep resumes: graphs and adj-lists that already exist are skipped, so a
# job that hits the time limit can be resubmitted unchanged and will continue.

#SBATCH --job-name=gamma-sweep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --account=torch_pr_929_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/gamma_sweep_%j.out

set -euo pipefail
mkdir -p slurm_out

# --- what to run ----------------------------------------------------------
DATASET=${DATASET:-glove25-25-angular}      # ann-benchmarks basename
METRIC=${METRIC:-euclidean}                 # label only, recorded in the CSV

R=${R:-32}
L=${L:-64}
ALPHA=${ALPHA:-1.0}

GAMMA_MIN=${GAMMA_MIN:-0.5}
GAMMA_MAX=${GAMMA_MAX:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.1}
SAMPLE_SIZE=${SAMPLE_SIZE:-}                # empty = binary default, 10 ln n

COV_MIN=${COV_MIN:-90};  COV_MAX=${COV_MAX:-100};  COV_STEP=${COV_STEP:-0.5}
EDGE_MIN=${EDGE_MIN:-1}; EDGE_MAX=${EDGE_MAX:-};   EDGE_STEP=${EDGE_STEP:-1}

SEARCH=${SEARCH:-1}                         # 1 = run the recall harness
SEARCH_K=${SEARCH_K:-"1 10"}   # space-separated; one harness run per k.
                               # k=100 aborts unless the graph can reach 100
                               # nodes from the start point (raise R first).

# --- paths ----------------------------------------------------------------
PROJECT=${PROJECT:-/scratch/pa2439/ANN-Search}
CODE=${CODE:-$PROJECT/navigable_graph_results/code}
DATA=${DATA:-$PROJECT/datasets}
OUT=${OUT:-$PROJECT/navigable_graph_results/gamma_sweep/$DATASET-R$R}

HDF5=$DATA/$DATASET.hdf5
BASE_FBIN=$DATA/$DATASET/base.fbin
QUERY_FBIN=$DATA/$DATASET/query.fbin

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PARLAY_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$OUT"

echo "=================================================================="
echo "job          $SLURM_JOB_ID on $(hostname)"
echo "dataset      $DATASET   (metric label: $METRIC)"
echo "build        R=$R L=$L alpha=$ALPHA"
echo "gamma grid   $GAMMA_MIN .. $GAMMA_MAX step $GAMMA_STEP"
echo "output       $OUT"
echo "=================================================================="

# --- sanity checks before burning an allocation ---------------------------
for f in "$HDF5" "$BASE_FBIN" "$QUERY_FBIN"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING: $f"
        echo "  (base/query fbin come from the HDF5 -> fbin step in Vamana-Runs.ipynb)"
        exit 1
    fi
done
for b in "$CODE/vamana/neighbors" "$CODE/../ParlayANN/algorithms/vamana/neighbors"; do
    if [[ ! -x "$b" ]]; then
        echo "MISSING BINARY: $b"
        echo "  build with: cd $(dirname "$b") && make"
        echo "  (on HPC prefer: make ARCHFLAGS= , so a login-node build runs on"
        echo "   compute nodes with a different CPU)"
        exit 1
    fi
done

# --- optional args --------------------------------------------------------
# Built as a plain string, not an array: it is interpolated into the quoted
# bash -c below, where "${arr[@]}" would not expand into separate arguments.
# Empty-array expansion under `set -u` is also a job-killer, so avoid arrays
# entirely. `if` rather than `[[ ]] &&` so a false test is not a `set -e` exit.
EXTRA=""
if [[ -n "$SAMPLE_SIZE" ]]; then EXTRA="$EXTRA --sample-size $SAMPLE_SIZE"; fi
if [[ -n "$EDGE_MAX"    ]]; then EXTRA="$EXTRA --edge-max $EDGE_MAX"; fi
if [[ "$SEARCH" == "1"  ]]; then EXTRA="$EXTRA --search --search-k $SEARCH_K"; fi  # unquoted: several k

# Large datasets: the per-source distance-row cache is O(n) floats per cached
# source, so it is only worth it well below a million points.
NPTS=$($SING /bin/bash -c "source /ext3/env.sh && conda activate big_ann && \
    python -c \"import numpy as np,sys; f=open('$BASE_FBIN','rb'); \
    print(int(np.fromfile(f,dtype=np.int32,count=2)[0]))\"")
echo "points: $NPTS"
if (( NPTS > 200000 )); then
    echo "large dataset -> --no-row-cache"
    EXTRA="$EXTRA --no-row-cache"
fi

srun $SING /bin/bash -c "
    source /ext3/env.sh && conda activate big_ann && \
    cd $CODE && \
    python gamma_sweep.py \
        --base-fbin  '$BASE_FBIN' \
        --query-fbin '$QUERY_FBIN' \
        --hdf5-path  '$HDF5' \
        --out-dir    '$OUT' \
        --dataset    '$DATASET' \
        --metric     '$METRIC' \
        --vamana-bin     '$CODE/../ParlayANN/algorithms/vamana/neighbors' \
        --mod-vamana-bin '$CODE/vamana/neighbors' \
        --R $R --L $L --alpha $ALPHA \
        --gamma-min $GAMMA_MIN --gamma-max $GAMMA_MAX --gamma-step $GAMMA_STEP \
        --cov-min $COV_MIN --cov-max $COV_MAX --cov-step $COV_STEP \
        --edge-min $EDGE_MIN --edge-step $EDGE_STEP \
        $EXTRA \
"

echo "=================================================================="
echo "done. results in $OUT"
ls -la "$OUT"/*.csv 2>/dev/null || true
