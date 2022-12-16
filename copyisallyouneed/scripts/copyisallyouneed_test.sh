#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method greedy
