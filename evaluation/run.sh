#!/bin/bash

cuda=$1
file_name=raw_files/enwiki_retro_result_sampling.json

# coherence
CUDA_VISIBLE_DEVICES=$cuda python coherence/test.py --test_path $file_name

# mauve
python mauve/test.py --test_path $file_name --device $cuda

# diversity
python diversity/test.py --test_path $file_name
