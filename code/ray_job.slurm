#!/bin/bash
# 9 nodes total: node 0 = Ray head + coordinator, nodes 1-8 = Ray workers
# Submit: sbatch ray_job.slurm

#SBATCH --job-name=nvgbl-ray
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --account=torch_pr_929_general
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm_out/slurm_%j_node%t.out

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32

# Node 0 is the head; derive its IP
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(ssh $HEAD_NODE hostname -I | awk '{print $2}')
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "All nodes: ${NODES[@]}"

# Start Ray head on node 0 and launch the coordinator script
srun --nodes=1 --ntasks=1 --nodelist=$HEAD_NODE $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    ray start --head \
        --node-ip-address=${HEAD_IP} \
        --port=${RAY_PORT} \
        --num-cpus=\$SLURM_CPUS_PER_TASK \
        --disable-usage-stats && \
    python /scratch/pa2439/ANN-Search/navigable_graph_results/code/distributed_robust_prune.py \
        --data $SCRATCH/ANN-Search/datasets/spacev1b \
        --save_path $SCRATCH/ANN-Search/navigable_graph_results \
        --num_points 10000 \
        --batch 100 \
        --num_shards 8 \
        --cpus 32 \
        --coordinator_cpus 32
" &

# Give head a moment to bind the port before workers connect
sleep 15

# Start Ray workers on nodes 1-8
WORKER_NODES=$(IFS=,; echo "${NODES[*]:1}")
srun --nodes=8 --ntasks=8 --nodelist=$WORKER_NODES $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    ray start \
        --address=${HEAD_IP}:${RAY_PORT} \
        --num-cpus=\$SLURM_CPUS_PER_TASK \
        --block
"

wait
