#!/bin/bash
# Submit: sbatch ray_job_hdf5.slurm

#SBATCH --job-name=nvgbl-ray-hdf5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --account=torch_pr_929_general
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=slurm_out/slurm_%j_node%t.out

# ---------------------------------------------------------------------------
# Dataset settings — edit before submitting
# ---------------------------------------------------------------------------
HDF5_FILE=$SCRATCH/ANN-Search/datasets/gist-960-euclidean.hdf5
HDF5_KEY=train
DATASET_NAME=gist
SAVE_PATH=$SCRATCH/ANN-Search/navigable_graph_results/new_results
NUM_POINTS=0       # 0 = all points
BATCH=50
NUM_SHARDS=2
CPUS=8
# ---------------------------------------------------------------------------

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS

# Node 0 is the head; derive its IP
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE=${NODES[0]}
HEAD_IP=$(ssh $HEAD_NODE hostname -I | awk '{print $2}')
RAY_PORT=6379

echo "Head node: $HEAD_NODE ($HEAD_IP)"
echo "All nodes: ${NODES[@]}"

# Start Ray head on node 0 and launch the coordinator script
srun --nodes=1 --ntasks=1 --cpu-bind=none --nodelist=$HEAD_NODE $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    ray start --head \
        --node-ip-address=${HEAD_IP} \
        --port=${RAY_PORT} \
        --num-cpus=\$SLURM_CPUS_PER_TASK \
        --disable-usage-stats && \
    python /scratch/pa2439/ANN-Search/navigable_graph_results/code/distributed_robust_prune_hdf5.py \
        --hdf5 ${HDF5_FILE} \
        --hdf5_key ${HDF5_KEY} \
        --name ${DATASET_NAME} \
        --save_path ${SAVE_PATH} \
        --num_points ${NUM_POINTS} \
        --batch ${BATCH} \
        --num_shards ${NUM_SHARDS} \
        --cpus ${CPUS} \
        --coordinator_cpus \$SLURM_CPUS_PER_TASK
" &

# Give head a moment to bind the port before workers connect
sleep 15

# Start Ray workers on nodes 1-8
WORKER_NODES=$(IFS=,; echo "${NODES[*]:1}")
srun --nodes=8 --ntasks=8 --cpu-bind=none --nodelist=$WORKER_NODES $SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    ray start \
        --address=${HEAD_IP}:${RAY_PORT} \
        --num-cpus=\$SLURM_CPUS_PER_TASK \
        --block
"

wait
