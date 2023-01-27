#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --random_seed 1.0 &
CUDA_VISIBLE_DEVICES=1 python test.py --random_seed 2.0 &
CUDA_VISIBLE_DEVICES=2 python test.py --random_seed 3.0 &
CUDA_VISIBLE_DEVICES=3 python test.py --random_seed 4.0 &
CUDA_VISIBLE_DEVICES=4 python test.py --random_seed 5.0 &
CUDA_VISIBLE_DEVICES=5 python test.py --random_seed 6.0 &
CUDA_VISIBLE_DEVICES=6 python test.py --random_seed 7.0 &
CUDA_VISIBLE_DEVICES=7 python test.py --random_seed 8.0 &

# greedy
# CUDA_VISIBLE_DEVICES=1 python test_greedy.py
