#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python gpt2_test.py --dataset wikitext103 --model gpt2
