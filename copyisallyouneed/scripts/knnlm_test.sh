#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python knnlm_test.py --dataset en_wiki --model knnlm --decoding_method nucleus_sampling
