#!/bin/bash

# wikitext103 index in wikitext test set (only the nucleus sampling)
# CUDA_VISIBLE_DEVICES=6 python knnlm_test_random.py --dataset wikitext103 --model knnlm --decoding_method nucleus_sampling &
# exit

# CUDA_VISIBLE_DEVICES=0 python knnlm_test_random.py --dataset en_wiki --model knnlm --decoding_method nucleus_sampling &

CUDA_VISIBLE_DEVICES=1 python knnlm_test.py --dataset lawmt --model knnlm --decoding_method greedy &
CUDA_VISIBLE_DEVICES=0 python knnlm_test.py --dataset en_wiki --model knnlm --decoding_method greedy &
exit

CUDA_VISIBLE_DEVICES=0 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 1.0 &
CUDA_VISIBLE_DEVICES=1 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 2.0 &
CUDA_VISIBLE_DEVICES=2 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 3.0 &
CUDA_VISIBLE_DEVICES=3 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 4.0 &
CUDA_VISIBLE_DEVICES=4 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 5.0 &
CUDA_VISIBLE_DEVICES=5 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 6.0 &
CUDA_VISIBLE_DEVICES=6 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 7.0 &
CUDA_VISIBLE_DEVICES=7 python knnlm_test_random.py --dataset lawmt --model knnlm --decoding_method nucleus_sampling --random_seed 8.0 &
