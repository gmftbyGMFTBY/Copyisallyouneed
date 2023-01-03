#!/bin/bash

# wikitext103
# <<COMMENT
python BARTScore/test.py --device 7 --test_path raw_files/gpt2_result.json &
python BARTScore/test.py --device 4 --test_path raw_files/gpt2_result_greedy.json &
python BARTScore/test.py --device 0 --test_path raw_files/copyisallyouneed_result.json &
python BARTScore/test.py --device 7 --test_path raw_files/copyisallyouneed_result_greedy.json &
python BARTScore/test.py --device 1 --test_path raw_files/neurlab_gpt2_result_greedy.json &
python BARTScore/test.py --device 4 --test_path raw_files/neurlab_gpt2_result_nucleus_sampling.json &
python BARTScore/test.py --device 2 --test_path raw_files/knnlm_result_greedy_full.json &
python BARTScore/test.py --device 7 --test_path raw_files/knnlm_result_nucleus_sampling_full.json &
python BARTScore/test.py --device 6 --test_path raw_files/retro_result_greedy.json &
python BARTScore/test.py --device 7 --test_path raw_files/retro_result_sampling.json &
# COMMENT

# lawmt
<<COMMENT
CUDA_VISIBLE_DEVICES=2 python mauve/test_roberta_v2.py --device 2 --test_path raw_files/lawmt_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=2 python mauve/test_roberta_v2.py --device 2 --test_path raw_files/lawmt_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=0 python mauve/test_roberta_v2.py --device 0 --test_path raw_files/lawmt_copyisallyouneed_result_greedy.json &
CUDA_VISIBLE_DEVICES=0 python mauve/test_roberta_v2.py --device 0 --test_path raw_files/lawmt_copyisallyouneed_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=1 python mauve/test_roberta_v2.py --device 1 --test_path raw_files/lawmt_neurlab_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=4 python mauve/test_roberta_v2.py --device 4 --test_path raw_files/lawmt_neurlab_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=6 python mauve/test_roberta_v2.py --device 6 --test_path raw_files/lawmt_knnlm_result_greedy_full.json &
CUDA_VISIBLE_DEVICES=7 python mauve/test_roberta_v2.py --device 7 --test_path raw_files/lawmt_knnlm_result_nucleus_sampling_full.json &
CUDA_VISIBLE_DEVICES=6 python mauve/test_roberta_v2.py --device 6 --test_path raw_files/lawmt_retro_result_greedy.json &
CUDA_VISIBLE_DEVICES=7 python mauve/test_roberta_v2.py --device 7 --test_path raw_files/lawmt_retro_result_sampling.json &
COMMENT


# en-wiki
<<COMMENT
CUDA_VISIBLE_DEVICES=2 python mauve/test_roberta_v2.py --device 2 --test_path raw_files/en_wiki_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=2 python mauve/test_roberta_v2.py --device 2 --test_path raw_files/en_wiki_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=0 python mauve/test_roberta_v2.py --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_greedy.json &
CUDA_VISIBLE_DEVICES=0 python mauve/test_roberta_v2.py --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=1 python mauve/test_roberta_v2.py --device 1 --test_path raw_files/en_wiki_neurlab_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=4 python mauve/test_roberta_v2.py --device 4 --test_path raw_files/en_wiki_neurlab_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=6 python mauve/test_roberta_v2.py --device 6 --test_path raw_files/enwiki_retro_result_greedy.json &
CUDA_VISIBLE_DEVICES=7 python mauve/test_roberta_v2.py --device 7 --test_path raw_files/enwiki_retro_result_sampling.json &
COMMENT
