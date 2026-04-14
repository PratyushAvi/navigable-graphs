#!/bin/bash
# Submit: sbatch ray_job_hdf5.slurm

#SBATCH --job-name=nvgbl-ray-hdf5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --account=torch_pr_929_general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=1:00:00
#SBATCH --output=slurm_out/slurm_%j.out

SIF=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=$SCRATCH/envs/overlay-15GB-500K.ext3
SING="singularity exec --fakeroot --overlay ${OVERLAY}:ro ${SIF}"

$SING /bin/bash -c "
    source /ext3/env.sh && conda activate big_ann && \
    python /scratch/pa2439/ANN-Search/navigable_graph_results/beam_search/beam_search.py \
        --adj_list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-glove25-euclidean.txt \
        --dataset /scratch/pa2439/ANN-Search/datasets/glove25-25-angular.hdf5 \
        --save_path /scratch/pa2439/ANN-Search/navigable_graph_results/ \
        --step_size 0.5 \
        --min_coverage 80 \
        --beam_widths 1 2 4 8 16 32 64 100 128 256
"
