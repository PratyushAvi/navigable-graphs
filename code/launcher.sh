singularity exec --fakeroot --overlay $SCRATCH/envs/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
source /ext3/env.sh
conda activate big_ann

MY_IP=$(hostname -I | awk '{print $2}')
ray start --head --node-ip-address=$MY_IP --port=6379 --num-cpus=16


python distributed_robust_prune.py --data=$SCRATCH/ANN-Search/datasets/spacev1b --num_shards=8 --cpus=32 --coordinator_cpus=16

/scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-mnist-euclidean.txt
/scratch/pa2439/ANN-Search/datasets/mnist-784-euclidean.hdf5

python beam_search.py --adj_list /scratch/pa2439/ANN-Search/navigable_graph_results/new_results/adj-list-mnist-euclidean.txt --dataset /scratch/pa2439/ANN-Search/datasets/mnist-784-euclidean.hdf5 --save_path /scratch/pa2439/ANN-Search/navigable_graph_results/new_results