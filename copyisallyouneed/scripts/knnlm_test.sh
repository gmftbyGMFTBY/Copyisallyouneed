#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python knnlm_test.py --dataset wikitext103 --model knnlm --decoding_method greedy
