#!/bin/bash

cuda=$1
# file_name=raw_files/wikitext103_copyisallyouneed_result_nucleus_sampling_v2.json
# file_name=raw_files/wikitext103_neurlab_gpt2_result_nucleus_sampling_v2.json
file_name=raw_files/en_wiki_knnlm_result_nucleus_sampling_full.json
# file_name=raw_files/en_wiki_knnlm_result_greedy_full.json

# coherence
CUDA_VISIBLE_DEVICES=$cuda python coherence/test.py --test_path $file_name

# mauve
# python mauve/test.py --test_path $file_name --device $cuda

# diversity
python diversity/test.py --test_path $file_name
