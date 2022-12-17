#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python knnlm_test.py --dataset lawmt --model knnlm --decoding_method greedy
