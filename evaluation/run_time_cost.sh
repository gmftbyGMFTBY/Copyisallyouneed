#!/bin/bash

python time_cost/test.py --test_path raw_files/wikitext_retro_result_greedy.json
exit

python time_cost/test.py --test_path raw_files/wikitext103_gpt2_result_nucleus_sampling.json
python time_cost/test.py --test_path raw_files/wikitext103_gpt2_result_greedy.json

# python time_cost/test.py --test_path raw_files/wikitext103_knnlm_result_nucleus_sampling_full.json
# python time_cost/test.py --test_path raw_files/wikitext103_knnlm_result_greedy_full.json


python time_cost/test.py --test_path raw_files/wikitext103_copyisallyouneed_result_nucleus_sampling.json
python time_cost/test.py --test_path raw_files/wikitext103_copyisallyouneed_result_greedy.json
