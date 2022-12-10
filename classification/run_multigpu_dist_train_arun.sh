#!/bin/bash
#SBATCH --job-name=2TeKIla
#SBATCH --output=./slurm_out_multig/%J.out
#SBATCH --partition=multigpu
#SBATCH --account=mbzuai
#SBATCH --ntasks=1
#SBATCH --gpus=16
#SBATCH --cpus-per-task=64

# #echo "Module = $(module avail)"

echo "Python Interpreter = $(which python)"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

NCCL_LL_THRESHOLD=0

echo "which nvcc = " $(which nvcc)

echo "nvcc --version = " $(nvcc --version)

#echo "NCCL_DEBUG = " $NCCL_DEBUG

echo "CUDA_VISIBLE_DEVICES = " $CUDA_VISIBLE_DEVICES

# echo "NCCL_LL_THRESHOLD = " $NCCL_LL_THRESHOLD


cd /nfs/users/ext_jose.viera/Next-ViT/classification

set -x

# export NCCL_LL_THRESHOLD=0
# export MKL_SERVICE_FORCE_INTEL=1

exp_name= nextvit_small_repro
time=$(date "+%m%d_%H%M%S")
save_root_dir=../${exp_name}/${time}

if [ ! -d ${save_root_dir} ]; then
    mkdir -p ${save_root_dir}
    echo save root dir is ${save_root_dir}.
else
    echo Error, save root dir ${save_roort_dir} exist, please run the shell again!
    exit 1
fi

/nfs/users/ext_jose.viera/condafiles/ENTER/envs/nextvit/bin/python \
    -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --output-dir ${save_root_dir} \
    --model nextvit_small \
    --batch-size 256 \
    --lr 5e-4 \
    --warmup-epochs 20 \
    --weight-decay 0.1 \
    --data-path /nfs/users/ext_jose.viera/datasets/imagenet \
    --dist-eval

