#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python gpt2_test.py --dataset wikitext103 --model gpt2 --decoding_method nucleus_sampling
