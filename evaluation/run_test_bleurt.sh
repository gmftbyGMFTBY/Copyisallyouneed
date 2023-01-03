#!/bin/bash

# wikitext-103
# <<COMMENT
CUDA_VISIBLE_DEVICES=6 python bleurt/test.py --test_path raw_files/gpt2_result.json &
CUDA_VISIBLE_DEVICES=0 python bleurt/test.py --test_path raw_files/gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=1 python bleurt/test.py --test_path raw_files/neurlab_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=7 python bleurt/test.py --test_path raw_files/neurlab_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=4 python bleurt/test.py --test_path raw_files/knnlm_result_greedy_full.json &
CUDA_VISIBLE_DEVICES=2 python bleurt/test.py --test_path raw_files/knnlm_result_nucleus_sampling_full.json &
CUDA_VISIBLE_DEVICES=6 python bleurt/test.py --test_path raw_files/retro_result_greedy.json &
CUDA_VISIBLE_DEVICES=7 python bleurt/test.py --test_path raw_files/retro_result_sampling.json &
CUDA_VISIBLE_DEVICES=2 python bleurt/test.py --test_path raw_files/copyisallyouneed_result.json &
CUDA_VISIBLE_DEVICES=4 python bleurt/test.py --test_path raw_files/copyisallyouneed_result_greedy.json &
# COMMENT
