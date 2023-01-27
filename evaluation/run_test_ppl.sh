#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 python perplexity/test.py --test_path raw_files/wikitext103_copyisallyouneed_result_nucleus_sampling.json
# CUDA_VISIBLE_DEVICES=2 python perplexity/test.py --test_path raw_files/wikitext103_copyisallyouneed_result_greedy.json

# CUDA_VISIBLE_DEVICES=2 python perplexity/test.py --test_path raw_files/wikitext103_gpt2_result_nucleus_sampling.json
# CUDA_VISIBLE_DEVICES=2 python perplexity/test.py --test_path raw_files/wikitext103_gpt2_result_greedy.json 

# CUDA_VISIBLE_DEVICES=2 python perplexity/test.py --test_path raw_files/knnlm_result_greedy_full.json &
# CUDA_VISIBLE_DEVICES=4 python perplexity/test.py --test_path raw_files/knnlm_result_nucleus_sampling_full.json &

# CUDA_VISIBLE_DEVICES=2 python perplexity/test.py --test_path raw_files/retro_result_greedy.json &
# CUDA_VISIBLE_DEVICES=4 python perplexity/test.py --test_path raw_files/retro_result_sampling.json &



# CUDA_VISIBLE_DEVICES=4 python perplexity/test.py --test_path raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_greedy_full_0.0_1.0.json &
# CUDA_VISIBLE_DEVICES=5 python perplexity/test.py --test_path raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_greedy_full_0.118_0.00785.json &
CUDA_VISIBLE_DEVICES=5 python perplexity/test.py --test_path raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_nucleus_sampling_full_0.118_0.00785.json &
