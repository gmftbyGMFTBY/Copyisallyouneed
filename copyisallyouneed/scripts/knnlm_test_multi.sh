#!/bin/bash

# wikitext103 index in wikitext test set (only the nucleus sampling)
# CUDA_VISIBLE_DEVICES=6 python knnlm_test_random.py --dataset wikitext103 --model knnlm --decoding_method nucleus_sampling &
# exit

# CUDA_VISIBLE_DEVICES=0 python knnlm_test_random.py --dataset en_wiki --model knnlm --decoding_method nucleus_sampling &

CUDA_VISIBLE_DEVICES=1 python knnlm_test.py --dataset lawmt --model knnlm --decoding_method greedy &
CUDA_VISIBLE_DEVICES=0 python knnlm_test.py --dataset en_wiki --model knnlm --decoding_method greedy &
