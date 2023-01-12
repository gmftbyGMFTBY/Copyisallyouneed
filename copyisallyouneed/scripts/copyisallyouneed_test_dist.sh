#!/bin/bash

for num in {0..15}
do
    cuda_device=$((num%8))
    echo "cuda device for worker $num is $cuda_device"
    CUDA_VISIBLE_DEVICES=cuda_device python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --worker_id $num&
done
