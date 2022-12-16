export NCCL_IB_DISABLE=1
# CUDA_VISIBLE_DEVICES=1,4,6,7 python train.py
CUDA_VISIBLE_DEVICES=0,1,2,4 python train.py
