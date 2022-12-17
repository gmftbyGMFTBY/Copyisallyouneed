#!/bin/bash

baseline_paths=(gpt2_result_nucleus_sampling.json neurlab_gpt2_result_nucleus_sampling.json retro_result_sampling.json knnlm_result_nucleus_sampling_full.json)
baseline_names=(gpt2_nucleus_sampling neurlab_gpt2_nucleus_sampling retro_nucleus_sampling knnlm_nucleus_sampling)

for i in "${!baseline_paths[@]}"; do
    baseline_path=raw_files/${baseline_paths[i]}
    baseline_name=${baseline_names[i]}
    echo "baseline path and name: $baseline_path; $baseline_name"
    rm -rf annotation_files/$baseline_name
    mkdir -p annotation_files/$baseline_name
    python make.py --baseline_path $baseline_path --baseline_name $baseline_name
    python generate.py --baseline_name $baseline_name
done
