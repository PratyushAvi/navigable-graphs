#!/bin/bash
#SBATCH --nodes=9
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=240GB
#SBATCH --job-name=nvgbl-mpi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --account=torch_pr_929_general

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

srun $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    python /path/to/mpi4py_distributed_robust_prune.py \
        --data /path/to/data \
        --num_points 10000 \
        --batch 50 \
        --num_shards 8 \
        --cpus 32
"
