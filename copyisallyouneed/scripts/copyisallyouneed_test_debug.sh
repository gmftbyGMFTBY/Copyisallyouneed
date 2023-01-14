#!/bin/bash


# CUDA_VISIBLE_DEVICES=0 python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling &

# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.1 &
CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test_debug.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.1 &
# CUDA_VISIBLE_DEVICES=0 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.2 &

# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test_debug.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.3
CUDA_VISIBLE_DEVICES=0 python copyisallyouneed_test_debug.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.3 &

# CUDA_VISIBLE_DEVICES=4 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.4 &
# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.5 &
# CUDA_VISIBLE_DEVICES=6 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.6 &
CUDA_VISIBLE_DEVICES=6 python copyisallyouneed_test_debug.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.6 &
# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.7 &
# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.8 &
# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 0.9 &

# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_subindex.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 1.0 &
CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_debug.py --dataset en_wiki --model copyisallyouneed --decoding_method nucleus_sampling --split_rate 1.0 &

# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test.py --dataset en_wiki --model copyisallyouneed --decoding_method greedy &
# CUDA_VISIBLE_DEVICES=0 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling
# CUDA_VISIBLE_DEVICES=1 python copyisallyouneed_test.py --dataset wikitext103 --model copyisallyouneed --decoding_method greedy &

# CUDA_VISIBLE_DEVICES=7 python copyisallyouneed_test_v2.py --dataset wikitext103 --model copyisallyouneed --decoding_method nucleus_sampling
