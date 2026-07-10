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
