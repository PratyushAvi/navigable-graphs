singularity exec --fakeroot --overlay $SCRATCH/envs/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
source /ext3/env.sh
conda activate big_ann

MY_IP=$(hostname -I | awk '{print $2}')
ray start --head --node-ip-address=$MY_IP --port=6379 --num-cpus=16


python distributed_robust_prune.py --data=$SCRATCH/ANN-Search/datasets/spacev1b --num_shards=8 --cpus=32 --coordinator_cpus=16

/scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-mnist-euclidean.txt
/scratch/pa2439/ANN-Search/datasets/mnist-784-euclidean.hdf5

python beam_search.py --adj_list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-mnist-euclidean.txt --dataset /scratch/pa2439/ANN-Search/datasets/mnist-784-euclidean.hdf5 --save_path /scratch/pa2439/ANN-Search/navigable_graph_results/

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-bigann.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/bigann-computed.txt \
    --dataset bigann \
    --metric euclidean \
    --total-points 1000000000 \
    --dimensions 128

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-yandex_deep.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/yandex_deep-computed.txt \
    --dataset yandex_deep \
    --metric euclidean \
    --total-points 1000000000 \
    --dimensions 96

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-facebook_sim_searchnet++.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/facebook_sim_searchnet++-computed.txt \
    --dataset facebook_sim_searchnet++ \
    --metric euclidean \
    --total-points 1000000000 \
    --dimensions 256

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-spacev1b-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/spacev1b-euclidean-computed.txt \
    --dataset spacev1b \
    --metric euclidean \
    --total-points 1402020720 \
    --dimensions 100

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-mnist-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/mnist-euclidean-computed.txt \
    --dataset mnist \
    --metric euclidean \
    --total-points 60000 \
    --dimensions 784

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-fashion_mnist-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/fashion_mnist-euclidean-computed.txt \
    --dataset fashion_mnist \
    --metric euclidean \
    --total-points 60000 \
    --dimensions 784

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-coco_i2i-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/coco_i2i-euclidean-computed.txt \
    --dataset coco_i2i \
    --metric euclidean \
    --total-points 113287 \
    --dimensions 512

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-coco_i2i-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/coco_i2i-euclidean-computed.txt \
    --dataset coco_i2i \
    --metric euclidean \
    --total-points 113287 \
    --dimensions 512

python updated_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-glove25-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/glove25-euclidean-computed.txt \
    --dataset glove25 \
    --metric euclidean \
    --total-points 1183514 \
    --dimensions 25

python edge_to_coverage_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-bigann.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/bigann-computed.txt \
    --dataset bigann \
    --metric euclidean \
    --total-points 1000000000 \
    --dimensions 128

python edge_to_coverage_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-yandex_deep.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/yandex_deep-computed.txt \
    --dataset yandex_deep \
    --metric euclidean \
    --total-points 1000000000 \
    --dimensions 96

python edge_to_coverage_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-facebook_sim_searchnet++.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/facebook_sim_searchnet++-computed.txt \
    --dataset facebook_sim_searchnet++ \
    --metric euclidean \
    --total-points 1000000000 \
    --dimensions 256

python edge_to_coverage_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-spacev1b-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/spacev1b-euclidean-computed.txt \
    --dataset spacev1b \
    --metric euclidean \
    --total-points 1402020720 \
    --dimensions 100

python extract_max_degree_curve.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-bigann.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/bigann-computed.txt \
    --total-points 1000000000 

python extract_max_degree_curve.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-yandex_deep.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/yandex_deep-computed.txt \
    --total-points 1000000000 

python extract_max_degree_curve.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-facebook_sim_searchnet++.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/facebook_sim_searchnet++-computed.txt \
    --total-points 1000000000 

python extract_max_degree_curve.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-spacev1b-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/spacev1b-euclidean-computed.txt \
    --total-points 1402020720 

python diagnose_coverage_drop.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-bigann.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/bigann-computed.txt \
    --total-points 1000000000 \
    --out /scratch/pa2439/ANN-Search/navigable_graph_results/edge_to_coverage/diag_bigann

python diagnose_coverage_drop.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-yandex_deep.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/yandex_deep-computed.txt \
    --total-points 1000000000 \
    --out /scratch/pa2439/ANN-Search/navigable_graph_results/edge_to_coverage/diag_yandex_deep

python diagnose_coverage_drop.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-facebook_sim_searchnet++.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/facebook_sim_searchnet++-computed.txt \
    --total-points 1000000000 \
    --out /scratch/pa2439/ANN-Search/navigable_graph_results/edge_to_coverage/diag_facebook_sim_searchnet++

python diagnose_coverage_drop.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-spacev1b-euclidean.txt \
    --computed /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/spacev1b-euclidean-computed.txt \
    --total-points 1402020720 \
    --out /scratch/pa2439/ANN-Search/navigable_graph_results/edge_to_coverage/diag_spacev1b

python buildgraphs.py \
    --dataset /scratch/pa2439/ANN-Search/datasets/mnist-784-euclidean.hdf5 \
    --gamma 0.99 \
    --delta 0.01 \
    --output /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/results/mnist_99p.csv

python buildgraphs.py \
    --dataset /scratch/pa2439/ANN-Search/datasets/glove25-25-angular.hdf5 \
    --gamma 0.99 \
    --delta 0.01 \
    --output /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/results/glove25_99p.csv

python buildgraphs.py \
    --dataset /scratch/pa2439/ANN-Search/datasets/coco_i2i-512-angular.hdf5 \
    --gamma 0.99 \
    --delta 0.01 \
    --output /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/results/coco_i2i_99p.csv

python beam_search_almost.py \
  --adj_list  /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/results/glove25_99p.csv \
  --dataset   /scratch/pa2439/ANN-Search/datasets/glove25-25-angular.hdf5 \
  --save_path /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/beam_results \
  --beam_widths 1 2 4 8 16 32 64 100 128 256 --tests 1000

python beam_search_almost.py \
  --adj_list  /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/results/mnist_99p.csv \
  --dataset   /scratch/pa2439/ANN-Search/datasets/mnist-784-euclidean.hdf5 \
  --save_path /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/beam_results \
  --beam_widths 1 2 4 8 16 32 64 100 128 256 --tests 1000

  python beam_search_almost.py \
  --adj_list  /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/results/coco_i2i_99p.csv \
  --dataset   /scratch/pa2439/ANN-Search/datasets/coco_i2i-512-angular.hdf5 \
  --save_path /scratch/pa2439/ANN-Search/navigable_graph_results/almost_navigable_graphs/beam_results \
  --beam_widths 1 2 4 8 16 32 64 100 128 256 --tests 1000

# mnist — 60,000 × 784
python coverage_to_degree_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-vamana-mnist-784-euclidean-R32-alpha1.txt \
    --dataset vamana-mnist-R32 --metric euclidean \
    --alpha 1.0 --method robust-prune \
    --total-points 60000 --dimensions 784

# fashion_mnist — 60,000 × 784
python coverage_to_degree_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-vamana-fashion_mnist-784-euclidean-R32-alpha1.txt \
    --dataset vamana-fashion_mnist-R32 --metric euclidean \
    --alpha 1.0 --method robust-prune \
    --total-points 60000 --dimensions 784

# coco_i2i — 113,287 × 512
python coverage_to_degree_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-vamana-coco_i2i-512-euclidean-R32-alpha1.txt \
    --dataset vamana-coco_i2i-R32 --metric euclidean \
    --alpha 1.0 --method robust-prune \
    --total-points 113287 --dimensions 512

# glove25 — 1,183,514 × 25
python coverage_to_degree_analysis.py \
    --adj-list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-vamana-glove25-25-euclidean-R32-alpha1.txt \
    --dataset vamana-glove25-R32 --metric euclidean \
    --alpha 1.0 --method robust-prune \
    --total-points 1183514 --dimensions 25

# =====================================================================
# gamma sweep: stock Vamana + modified Vamana over a gamma grid, then
# coverage adj-lists, stats, and ParlayANN's recall harness.
#
# Build both binaries first:
#     cd code/vamana                        && make
#     cd ParlayANN/algorithms/vamana        && make
#
# Each job resumes: graphs and adj-lists already present are skipped, so a
# job that hits the time limit can be resubmitted unchanged. Results for a
# dataset accumulate into one CSV pair keyed by (method, gamma, alpha, R, L, S),
# so these can also be rerun with a different R or gamma grid and the rows
# coexist rather than overwrite.
#
# DATASET must match the hdf5 basename; base.fbin/query.fbin come from the
# HDF5 -> fbin step in Vamana-Runs.ipynb.
#
# DATASET is the hdf5 basename, so coco_i2i and glove25 keep their "-angular"
# filenames; everything here is built and searched with euclidean distance, which
# is what METRIC records.
#
# OUT is set per dataset. Runs sharing an OUT accumulate into the one CSV pair
# there, keyed by (method, gamma, alpha, R, L, S), so a rerun with another gamma
# grid or sample size adds rows instead of starting a new file. A different R
# gets its own directory, since its graphs and adj-lists differ.
# =====================================================================

# mnist — 60,000 x 784
sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/mnist-R64,DATASET=mnist-784-euclidean,METRIC=euclidean,R=64,L=64,GAMMA_MIN=0.9,GAMMA_MAX=1.0,GAMMA_STEP=0.1 \
    gamma_sweep.slurm

# fashion_mnist — 60,000 x 784
sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/fashion_mnist-R64,DATASET=fashion_mnist-784-euclidean,METRIC=euclidean,R=64,L=64,GAMMA_MIN=0.9,GAMMA_MAX=1.0,GAMMA_STEP=0.1 \
    gamma_sweep.slurm

# coco_i2i — 113,287 x 512
sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/coco_i2i-R64,DATASET=coco_i2i-512-angular,METRIC=euclidean,R=64,L=64,GAMMA_MIN=0.9,GAMMA_MAX=1.0,GAMMA_STEP=0.1 \
    gamma_sweep.slurm

# glove25 — 1,183,514 x 25. The adj-list pass dominates here, so give it the
# full time limit and skip the row cache (the slurm script does that above
# 200k points automatically).
sbatch --time=48:00:00 \
    --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/glove25-R64,DATASET=glove25-25-angular,METRIC=euclidean,R=64,L=64,GAMMA_MIN=0.9,GAMMA_MAX=1.0,GAMMA_STEP=0.1 \
    gamma_sweep.slurm

# --- variations -------------------------------------------------------
# Graphs only, no coverage pass or stats (fast; see degree/edge counts first):
# sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/mnist-R64,DATASET=mnist-784-euclidean,SEARCH=0 gamma_sweep.slurm
#
# A finer gamma grid near 1.0:
# sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/mnist-R64,DATASET=mnist-784-euclidean,GAMMA_MIN=0.9,GAMMA_MAX=1.0,GAMMA_STEP=0.02 gamma_sweep.slurm
#
# A different sample size (S is part of the key, so it will not overwrite):
# sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/mnist-R64,DATASET=mnist-784-euclidean,SAMPLE_SIZE=500 gamma_sweep.slurm
#
# A different max degree:
# sbatch --export=ALL,OUT=/scratch/pa2439/ANN-Search/navigable_graph_results/gamma_sweep/mnist-R128,DATASET=mnist-784-euclidean,R=128,L=256 gamma_sweep.slurm
