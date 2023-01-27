#!/bin/bash

# exec_file=copyisallyouneed_test_subindex_random.py
exec_file=copyisallyouneed_test.py

CUDA_VISIBLE_DEVICES=7 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy &
exit


CUDA_VISIBLE_DEVICES=0 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.0 &
CUDA_VISIBLE_DEVICES=1 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.001 &
CUDA_VISIBLE_DEVICES=6 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.003 &
CUDA_VISIBLE_DEVICES=7 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.01 &

exit

CUDA_VISIBLE_DEVICES=1 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.03 &
CUDA_VISIBLE_DEVICES=0 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.1 &
CUDA_VISIBLE_DEVICES=6 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 0.3 &
CUDA_VISIBLE_DEVICES=7 python $exec_file --dataset en_wiki --model copyisallyouneed --decoding_method greedy --split_rate 1.0 &
