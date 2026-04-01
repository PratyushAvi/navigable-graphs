#!/bin/bash
# Heterogeneous job: group 0 = head/coordinator (1 node), group 1 = workers (8 nodes)
# Submit: sbatch ray_job.slurm

#SBATCH --job-name=nvgbl-ray
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --account=torch_pr_929_general

# --- Head node (coordinator) ---
#SBATCH hetjob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/slurm_%j_coordinator.out

# --- Worker nodes ---
#SBATCH hetjob
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/slurm_%j_worker%t.out

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

# Get the IP of the head node (het-group 0, task 0)
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0" | head -n1)
HEAD_IP=$(getent hosts $HEAD_NODE | awk '{print $1}')
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_IP)"

# Start Ray head on group 0
srun --het-group=0 $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    ray start --head \
        --node-ip-address=${HEAD_IP} \
        --port=${RAY_PORT} \
        --num-cpus=\$SLURM_CPUS_PER_TASK \
        --disable-usage-stats && \
    python /scratch/pa2439/ANN-Search/navigable_graph_results/code/distributed_robust_prune.py \
        --data $SCRATCH/ANN-Search/datasets/spacev1b \
        --num_points 10000 \
        --batch 50 \
        --num_shards 8 \
        --cpus 32 \
        --coordinator_cpus 32
" &

# Give head a moment to bind the port before workers connect
sleep 15

# Start Ray workers on group 1
srun --het-group=1 $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    ray start \
        --address=${HEAD_IP}:${RAY_PORT} \
        --num-cpus=\$SLURM_CPUS_PER_TASK \
        --block
"

wait
