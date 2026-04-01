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
#SBATCH --output=slurm_out/slurm_%j_rank%t.out
#SBATCH --account=torch_pr_929_general

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Build hostfile: one line per node, one slot per node (one MPI rank per node)
scontrol show hostnames $SLURM_JOB_NODELIST | awk '{print $0 " slots=1"}' > hostfile_$SLURM_JOB_ID

$SING /bin/bash -c "
    source /ext3/miniforge3/etc/profile.d/conda.sh && \
    conda activate big_ann && \
    mpirun -np 9 \
        --hostfile hostfile_$SLURM_JOB_ID \
        --map-by node \
        --mca plm rsh \
        --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no" \
        -x OMP_NUM_THREADS \
        -x OPENBLAS_NUM_THREADS \
        -x OMPI_ALLOW_RUN_AS_ROOT \
        -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM \
        python /scratch/pa2439/ANN-Search/navigable_graph_results/code/mpi4py_distributed_robust_prune.py \
            --data $SCRATCH/ANN-Search/datasets/spacev1b \
            --num_points 10000 \
            --batch 50 \
            --num_shards 8 \
            --cpus 32
"
