#!/bin/bash

# en-wiki index for en-wiki test set
CUDA_VISIBLE_DEVICES=7 python gpt2_test.py --dataset wikitext103 --model gpt2 --decoding_method greedy &

CUDA_VISIBLE_DEVICES=1 python gpt2_test_random.py --dataset en_wiki --model gpt2 --decoding_method nucleus_sampling &
# wikitext103 index for en-wiki test set
# CUDA_VISIBLE_DEVICES=4 python gpt2_test_random.py --dataset wikitext103 --model gpt2 --decoding_method nucleus_sampling &
