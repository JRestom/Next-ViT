#!/bin/bash
#SBATCH --exclude=p3-r52-a.g42cloud.net
#SBATCH --exclude=p4-r68-a.g42cloud.net
#SBATCH --exclude=p4-r67-b.g42cloud.net
#SBATCH --job-name=VL-LTR
#SBATCH --output=./slurm_out_multig/%J.out
#SBATCH --partition=multigpu
#SBATCH --account=mbzuai

#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64


set -x
GPUS=$1

exp_name=nextvit_cls_exp
time=$(date "+%m%d_%H%M%S")
save_root_dir=../${exp_name}/${time}

if [ ! -d ${save_root_dir} ]; then
    mkdir -p ${save_root_dir}
    echo save root dir is ${save_root_dir}.
else
    echo Error, save root dir ${save_roort_dir} exist, please run the shell again!
    exit 1
fi


python3 -m torch.distributed.launch --nproc_per_node=$GPUS --use_env main.py \
--output-dir ${save_root_dir} \
--dist-eval ${@:2}


CONFIG=$1
GPUS=$2
CPUS=$[GPUS*4]
PORT=${PORT:-8666}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi

CONFIG_NAME=${CONFIG##*/}
CONFIG_NAME=${CONFIG_NAME%.*}

OUTPUT_DIR="./checkpoints/${CONFIG_NAME}"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p ${OUTPUT_DIR}
fi

/nfs/users/ext_jose.viera/condafiles/ENTER/envs/nextvit/bin/python \
    -m torch.distributed.launch --nproc_per_node=$GPUS main.py \
    --port=$PORT \
    --num_workers 4 \
    --config $CONFIG ${@:3} \
    --resume "./checkpoints/${CONFIG_NAME}/checkpoint.pth" \
    2>&1 | tee -a ${OUTPUT_DIR}/train.log

