#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=nvgbl-workers
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_out/slurm_%j.out
#SBATCH --account=torch_pr_929_general
HEAD_ADDR=$1   # passed in as sbatch argument, e.g. "10.x.x.x:6379"

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

srun $SING /bin/bash -c "
    /ext3/miniforge3/envs/big_ann/bin/ray start \
        --address=$HEAD_ADDR \
        --num-cpus=$SLURM_CPUS_PER_TASK --block
"
