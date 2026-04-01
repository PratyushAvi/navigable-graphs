#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=240GB
#SBATCH --job-name=nvgbl-zmq-workers
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_out/slurm_%j_worker%t.out
#SBATCH --account=torch_pr_929_general

HEAD_IP=$1   # e.g.: sbatch zmq_workers.slurm 10.32.36.186

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

srun $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    python /scratch/pa2439/ANN-Search/navigable_graph_results/code/zmq_distributed_robust_prune.py worker \
        --shard_id \$SLURM_PROCID \
        --coordinator ${HEAD_IP}:5555 \
        --data $SCRATCH/ANN-Search/datasets/spacev1b \
        --num_shards 8 \
        --cpus 32
"
