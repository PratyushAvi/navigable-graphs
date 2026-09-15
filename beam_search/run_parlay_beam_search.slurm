#!/bin/bash
# ParlayANN's recall harness on this project's robust-prune / set-cover graphs.
#
# Submit:          sbatch run_parlay_beam_search.slurm
# Other dataset:   sbatch --export=ALL,DATASET=glove25-25-angular,ADJ=... run_parlay_beam_search.slurm
# Coverage sweep:  sbatch --export=ALL,COV_MIN=99,COV_MAX=100,COV_STEP=0.25 run_parlay_beam_search.slurm

#SBATCH --job-name=parlay-beam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --account=torch_pr_929_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=slurm_out/parlay_beam_%j.out

set -euo pipefail
mkdir -p slurm_out

DATASET=${DATASET:-mnist-784-euclidean}
NAME=${NAME:-mnist}                       # short name used in filenames
METHOD=${METHOD:-robust-prune}
K=${K:-10}

PROJECT=${PROJECT:-/scratch/pa2439/ANN-Search}
DATA=${DATA:-$PROJECT/datasets}
RESULTS=${RESULTS:-$PROJECT/navigable_graph_results}
BEAM=${BEAM:-$RESULTS/beam_search}
OUT=${OUT:-$RESULTS/parlay_search/$NAME}

ADJ=${ADJ:-$RESULTS/new_results/adj-list-$NAME-euclidean.txt}
COMPUTED=${COMPUTED:-$RESULTS/new_results/$NAME-euclidean-computed.txt}
BASE_FBIN=${BASE_FBIN:-$DATA/$DATASET/base.fbin}
QUERY_FBIN=${QUERY_FBIN:-$DATA/$DATASET/query.fbin}
HDF5=${HDF5:-$DATA/$DATASET.hdf5}

# Coverage sweep: leave COV_MIN empty to search the whole graph only.
COV_MIN=${COV_MIN:-}
COV_MAX=${COV_MAX:-100}
COV_STEP=${COV_STEP:-0.5}

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PARLAY_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p "$OUT"

echo "=================================================================="
echo "job      $SLURM_JOB_ID on $(hostname)"
echo "dataset  $DATASET   method $METHOD   k=$K"
echo "adj-list $ADJ"
echo "output   $OUT"
echo "=================================================================="

for f in "$ADJ" "$BASE_FBIN" "$QUERY_FBIN" "$HDF5"; do
    if [[ ! -f "$f" ]]; then echo "MISSING: $f"; exit 1; fi
done

BIN=$RESULTS/ParlayANN/algorithms/vamana/neighbors
if [[ ! -x "$BIN" ]]; then
    echo "MISSING BINARY: $BIN"
    echo "  build with: cd $(dirname "$BIN") && make ARCHFLAGS="
    exit 1
fi

# Bare-neighbourhood adjacency files need their computed-sources file; prefixed
# ones do not have a separate file at all.
EXTRA=""
if [[ -f "$COMPUTED" ]]; then
    echo "using computed-sources file: $COMPUTED"
    EXTRA="$EXTRA --computed $COMPUTED"
fi
if [[ -n "$COV_MIN" ]]; then
    EXTRA="$EXTRA --coverage-range $COV_MIN $COV_MAX $COV_STEP"
fi

srun $SING /bin/bash -c "
    source /ext3/env.sh && conda activate big_ann && \
    cd $BEAM && \
    python parlay_beam_search.py \
        --adj-list   '$ADJ' \
        --base-fbin  '$BASE_FBIN' \
        --query-fbin '$QUERY_FBIN' \
        --hdf5       '$HDF5' \
        --out-dir    '$OUT' \
        --dataset    '$NAME' \
        --method     '$METHOD' \
        --vamana-bin '$BIN' \
        --k $K \
        $EXTRA \
"

echo "=================================================================="
echo "done. results in $OUT"
ls -la "$OUT"/*.csv 2>/dev/null || true
