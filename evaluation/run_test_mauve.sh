#!/bin/bash

evaluate_file_name=mauve/test.py

# wikitext103

<<COMMENT
python $evaluate_file_name --device 6 --test_path raw_files/gpt2_result_greedy.json &
python $evaluate_file_name --device 0 --test_path raw_files/gpt2_result.json &
python $evaluate_file_name --device 1 --test_path raw_files/copyisallyouneed_result.json &
python $evaluate_file_name --device 7 --test_path raw_files/copyisallyouneed_result_greedy.json &
# python $evaluate_file_name --device 1 --test_path raw_files/neurlab_gpt2_result_greedy.json &
# python $evaluate_file_name --device 4 --test_path raw_files/neurlab_gpt2_result_nucleus_sampling.json &
python $evaluate_file_name --device 2 --test_path raw_files/knnlm_result_greedy_full.json &
python $evaluate_file_name --device 4 --test_path raw_files/knnlm_result_nucleus_sampling_full.json &
python $evaluate_file_name --device 0 --test_path raw_files/retro_result_greedy.json &
python $evaluate_file_name --device 6 --test_path raw_files/retro_result_sampling.json &
COMMENT



<<COMMENT
CUDA_VISIBLE_DEVICES=2 python $evaluate_file_name --device 2 --test_path raw_files/gpt2_result.json &
CUDA_VISIBLE_DEVICES=6 python $evaluate_file_name --device 6 --test_path raw_files/gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=0 python $evaluate_file_name --device 0 --test_path raw_files/copyisallyouneed_result.json &
CUDA_VISIBLE_DEVICES=7 python $evaluate_file_name --device 7 --test_path raw_files/copyisallyouneed_result_greedy.json &
# CUDA_VISIBLE_DEVICES=1 python $evaluate_file_name --device 1 --test_path raw_files/neurlab_gpt2_result_greedy.json &
# CUDA_VISIBLE_DEVICES=4 python $evaluate_file_name --device 4 --test_path raw_files/neurlab_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=1 python $evaluate_file_name --device 1 --test_path raw_files/knnlm_result_greedy_full.json &
CUDA_VISIBLE_DEVICES=4 python $evaluate_file_name --device 4 --test_path raw_files/knnlm_result_nucleus_sampling_full.json &
CUDA_VISIBLE_DEVICES=0 python $evaluate_file_name --device 0 --test_path raw_files/retro_result_greedy.json &
CUDA_VISIBLE_DEVICES=6 python $evaluate_file_name --device 6 --test_path raw_files/retro_result_sampling.json &
COMMENT

# lawmt

<<COMMENT
python $evaluate_file_name --device 6 --test_path raw_files/lawmt_gpt2_result_greedy_v2.json &
python $evaluate_file_name --device 7 --test_path raw_files/lawmt_gpt2_result_nucleus_sampling_v2.json &
# python $evaluate_file_name --device 0 --test_path raw_files/lawmt_copyisallyouneed_result_greedy.json &
# python $evaluate_file_name --device 0 --test_path raw_files/lawmt_copyisallyouneed_result_nucleus_sampling.json &
# python $evaluate_file_name --device 1 --test_path raw_files/lawmt_neurlab_gpt2_result_greedy.json &
# python $evaluate_file_name --device 4 --test_path raw_files/lawmt_neurlab_gpt2_result_nucleus_sampling.json &
# python $evaluate_file_name --device 6 --test_path raw_files/lawmt_knnlm_result_greedy_full.json &
# python $evaluate_file_name --device 7 --test_path raw_files/lawmt_knnlm_result_nucleus_sampling_full.json &
# python $evaluate_file_name --device 6 --test_path raw_files/lawmt_retro_result_greedy.json &
# python $evaluate_file_name --device 7 --test_path raw_files/lawmt_retro_result_sampling.json &
COMMENT


# python $evaluate_file_name --device 1 --test_path raw_files/lawmt_knnlm_result_nucleus_sampling_full.json &
# python $evaluate_file_name --device 6 --test_path raw_files/lawmt_knnlm_result_greedy_full.json &
# python $evaluate_file_name --device 7 --test_path raw_files/lawmt_retro_result_sampling.json &
# python $evaluate_file_name --device 6 --test_path raw_files/lawmt_retro_result_greedy.json &
# exit




<<COMMENT
CUDA_VISIBLE_DEVICES=2 python $evaluate_file_name --device 2 --test_path raw_files/lawmt_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=2 python $evaluate_file_name --device 2 --test_path raw_files/lawmt_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=0 python $evaluate_file_name --device 0 --test_path raw_files/lawmt_copyisallyouneed_result_greedy.json &
CUDA_VISIBLE_DEVICES=0 python $evaluate_file_name --device 0 --test_path raw_files/lawmt_copyisallyouneed_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=1 python $evaluate_file_name --device 1 --test_path raw_files/lawmt_neurlab_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=4 python $evaluate_file_name --device 4 --test_path raw_files/lawmt_neurlab_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=6 python $evaluate_file_name --device 6 --test_path raw_files/lawmt_knnlm_result_greedy_full.json &
CUDA_VISIBLE_DEVICES=7 python $evaluate_file_name --device 7 --test_path raw_files/lawmt_knnlm_result_nucleus_sampling_full.json &
CUDA_VISIBLE_DEVICES=6 python $evaluate_file_name --device 6 --test_path raw_files/lawmt_retro_result_greedy.json &
CUDA_VISIBLE_DEVICES=7 python $evaluate_file_name --device 7 --test_path raw_files/lawmt_retro_result_sampling.json &
COMMENT


# en-wiki
# <<COMMENT
# python $evaluate_file_name --device 2 --test_path raw_files/en_wiki_gpt2_result_greedy_v2.json &
# python $evaluate_file_name --device 1 --test_path raw_files/en_wiki_gpt2_result_nucleus_sampling_v2.json &
# python $evaluate_file_name --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_greedy.json &
# python $evaluate_file_name --device 3 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling.json &
# python $evaluate_file_name --device 1 --test_path raw_files/en_wiki_neurlab_gpt2_result_greedy.json &
# python $evaluate_file_name --device 4 --test_path raw_files/en_wiki_neurlab_gpt2_result_nucleus_sampling.json &
# python $evaluate_file_name --device 6 --test_path raw_files/enwiki_retro_result_greedy.json &
# python $evaluate_file_name --device 7 --test_path raw_files/enwiki_retro_result_sampling.json &
# python $evaluate_file_name --device 6 --test_path raw_files/en_wiki_knnlm_result_greedy_full.json &
# python $evaluate_file_name --device 7 --test_path raw_files/en_wiki_knnlm_result_nucleus_sampling_full.json &
# COMMENT

<<COMMENT
CUDA_VISIBLE_DEVICES=2 python mauve/test_roberta_v2.py --device 2 --test_path raw_files/en_wiki_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=2 python mauve/test_roberta_v2.py --device 2 --test_path raw_files/en_wiki_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=0 python mauve/test_roberta_v2.py --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_greedy.json &
CUDA_VISIBLE_DEVICES=0 python mauve/test_roberta_v2.py --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=1 python mauve/test_roberta_v2.py --device 1 --test_path raw_files/en_wiki_neurlab_gpt2_result_greedy.json &
CUDA_VISIBLE_DEVICES=4 python mauve/test_roberta_v2.py --device 4 --test_path raw_files/en_wiki_neurlab_gpt2_result_nucleus_sampling.json &
CUDA_VISIBLE_DEVICES=6 python mauve/test_roberta_v2.py --device 6 --test_path raw_files/enwiki_retro_result_greedy.json &
CUDA_VISIBLE_DEVICES=7 python mauve/test_roberta_v2.py --device 7 --test_path raw_files/enwiki_retro_result_sampling.json &
CUDA_VISIBLE_DEVICES=6 python mauve/test_roberta_v2.py --device 6 --test_path raw_files/en_wiki_knnlm_result_greedy_full.json &
CUDA_VISIBLE_DEVICES=7 python mauve/test_roberta_v2.py --device 7 --test_path raw_files/en_wiki_knnlm_result_nucleus_sampling_full.json &
COMMENT

# python $evaluate_file_name --device 0 --test_path raw_files/en_wiki_gpt2_result_nucleus_sampling.json &
# python $evaluate_file_name --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.0_nprobe_10.json &
# python $evaluate_file_name --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.1_nprobe_10.json &
# python $evaluate_file_name --device 2 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.2_nprobe_10.json &
# python $evaluate_file_name --device 1 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.3_nprobe_10.json &
# python $evaluate_file_name --device 6 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.4_nprobe_10.json &
# python $evaluate_file_name --device 2 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.5_nprobe_10.json &
# python $evaluate_file_name --device 4 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.6_nprobe_10.json &
# python $evaluate_file_name --device 4 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.7_nprobe_10.json &
# python $evaluate_file_name --device 0 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.8_nprobe_10.json &
# python $evaluate_file_name --device 3 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_0.9_nprobe_10.json &
# python $evaluate_file_name --device 5 --test_path raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_1.0_nprobe_10.json &


# python mauve/test_multi.py --device 0 --split_rate 1.0 &
# python mauve/test_multi.py --device 1 --split_rate 0.3 &
# python mauve/test_multi.py --device 0 --split_rate 0.1 &
# python mauve/test_multi.py --device 1 --split_rate 0.03 &

# python mauve/test_multi.py --device 6 --split_rate 0.01 &
# python mauve/test_multi.py --device 7 --split_rate 0.003 &
# python mauve/test_multi.py --device 6 --split_rate 0.001 &
# python mauve/test_multi.py --device 7 --split_rate 0.0 &

# python mauve/test_multi_gpt2.py --device 1 --dataset wikitext103 &
# python mauve/test_multi_gpt2.py --device 1 --dataset wikitext103 &

# python mauve/test.py --device 2 --test_path raw_files/wikitext103_gpt2_result_greedy.json
# python mauve/test.py --device 1 --test_path raw_files/wikitext103_knnlm_result_greedy_full.json
# python mauve/test.py --device 1 --test_path raw_files/random_runs_en_wiki/en_wiki_knnlm_result_nucleus_sampling_on_wikitext103_index_wikitext103_testset_seed_1.0.json
# python mauve/test.py --device 2 --test_path raw_files/random_runs_en_wiki/en_wiki_knnlm_result_nucleus_sampling_on_wikitext103_index_wikitext103_testset_seed_5.0.json

# python mauve/test.py --device 7 --test_path raw_files/knnlm_wikitext103/wikitext103_knnlm_result_nucleus_sampling_on_wikitext103_index_wikitext103_testset_seed_1.0.json &
# python mauve/test.py --device 6 --test_path raw_files/knnlm_wikitext103/wikitext103_knnlm_result_nucleus_sampling_on_wikitext103_index_wikitext103_testset_seed_5.0.json &
# python mauve/test.py --device 6 --test_path raw_files/random_runs_retro/en_wiki_retro_result_greedy_en_wiki_index.json &

# python mauve/test.py --device 6 --test_path raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_nucleus_sampling_full_0.0_1.0.json &
# python mauve/test.py --device 6 --test_path raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_nucleus_sampling_full_0.118_0.00785.json &
# python mauve/test.py --device 7 --test_path raw_files/random_runs_wikitext103_testset_knnlm/wikitext103_knnlm_result_greedy_full_0.118_0.00785.json &

python mauve/test.py --device 1 --test_path raw_files/random_runs_lawmt/en_wiki_knnlm_result_greedy_full_0.118_0.00785.json &
python mauve/test.py --device 0 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_greedy_full_0.118_0.00785.json &
exit


python mauve/test.py --device 0 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_1.0.json &
python mauve/test.py --device 1 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_2.0.json &
python mauve/test.py --device 2 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_3.0.json &
python mauve/test.py --device 3 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_4.0.json &
python mauve/test.py --device 4 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_5.0.json &
python mauve/test.py --device 5 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_6.0.json &
python mauve/test.py --device 6 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_7.0.json &
python mauve/test.py --device 7 --test_path raw_files/random_runs_lawmt/lawmt_knnlm_result_nucleus_sampling_seed_8.0.json &
