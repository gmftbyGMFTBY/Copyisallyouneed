#!/bin/bash

# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method greedy
CUDA_VISIBLE_DEVICES=6 python copyisallyouneed_test.py --dataset lawmt --model copyisallyouneed --decoding_method greedy 
