#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python knnlm_test.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling &
CUDA_VISIBLE_DEVICES=1 python knnlm_test.py --dataset lawmt --model knnlm --decoding_method greedy &
