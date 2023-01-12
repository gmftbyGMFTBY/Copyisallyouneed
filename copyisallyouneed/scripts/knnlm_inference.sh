#!/bin/bash
export NCCL_IB_DISABLE=1

cuda=$1
gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28204 inference.py \
    --dataset lawmt\
    --model knnlm
