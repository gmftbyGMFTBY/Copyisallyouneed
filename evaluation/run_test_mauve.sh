#!/bin/bash

# mauve
python mauve/test_v2.py --device 0 --test_path raw_files/gpt2_result.json > 1.txt &
python mauve/test_v2.py --device 1 --test_path raw_files/neurlab_gpt2_result_nucleus_sampling.json > 2.txt &
python mauve/test_v2.py --device 2 --test_path raw_files/knnlm_result.json > 3.txt &
python mauve/test_v2.py --device 6 --test_path raw_files/retro_result_sampling.json > 4.txt &
python mauve/test_v2.py --device 4 --test_path raw_files/copyisallyouneed_result.json > 5.txt &
