#!/bin/bash


CUDA_VISIBLE_DEVICES=4 python copyisallyouneed_test_debug.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 1.0
