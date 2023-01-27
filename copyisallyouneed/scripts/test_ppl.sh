#!/bin/bash

model=$1
CUDA_VISIBLE_DEVICES=7 python test_ppl.py --dataset wikitext103 --model $model
