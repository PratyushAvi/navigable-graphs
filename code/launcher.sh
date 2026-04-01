singularity exec --fakeroot --overlay $SCRATCH/envs/overlay-15GB-500K.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash
source /ext3/env.sh
conda activate big_ann

MY_IP=$(hostname -I | awk '{print $1}')
ray start --head --node-ip-address=$MY_IP --port=6379 --num-cpus=16


python distributed_robust_prune.py --data=$SCRATCH/ANN-Search/datasets/spacev1b --num_shards=8 --cpus=32 --coordinator_cpus=16

