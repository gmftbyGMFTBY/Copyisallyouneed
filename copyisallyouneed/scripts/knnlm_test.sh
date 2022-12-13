#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python knnlm_test.py --dataset wikitext103 --model knnlm 
