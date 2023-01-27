#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python knnlm_test.py --dataset wikitext103 --model knnlm --decoding_method nucleus_sampling &
CUDA_VISIBLE_DEVICES=6 python knnlm_test.py --dataset wikitext103 --model knnlm --decoding_method greedy &
