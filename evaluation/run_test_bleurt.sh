#!/bin/bash

# mauve
CUDA_VISIBLE_DEVICES=1 python bleurt/test_official.py --test_path raw_files/gpt2_result.json
CUDA_VISIBLE_DEVICES=1 python bleurt/test_official.py --test_path raw_files/copyisallyouneed_result.json
