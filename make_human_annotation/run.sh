#!/bin/bash

# baseline_paths=(gpt2_result_nucleus_sampling.json neurlab_gpt2_result_nucleus_sampling.json retro_result_sampling.json knnlm_result_nucleus_sampling_full.json lawmt_gpt2_nucleus_sampling.json en_wiki_gpt2_nucleus_sampling.json lawmt_copysiallyouneed_nucleus_sampling.json en_wiki_copyisallyouneed_nucleus_sampling.json)
# baseline_names=(gpt2_nucleus_sampling neurlab_gpt2_nucleus_sampling retro_nucleus_sampling knnlm_nucleus_sampling lawmt_gpt2_nucleus_sampling en_wiki_gpt2_nucleus_sampling lawmt_copyisallyouneed_nucleus_sampling en_wiki_copyisallyouneed_nucleus_sampling)

# baseline_paths=(lawmt_gpt2_result_nucleus_sampling.json)
# baseline_names=(lawmt_gpt2_nucleus_sampling)

# baseline_paths=(en_wiki_gpt2_result_nucleus_sampling.json)
# baseline_names=(en_wiki_gpt2_nucleus_sampling_on_en_wiki)

baseline_paths=(en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_en_wiki_testset_0.7_nprobe_10.json)
baseline_names=(en_wiki_copyisallyouneed_nucleus_sampling_on_en_wiki)

for i in "${!baseline_paths[@]}"; do
    baseline_path=raw_files/${baseline_paths[i]}
    baseline_name=${baseline_names[i]}
    echo "baseline path and name: $baseline_path; $baseline_name"
    rm -rf annotation_files/$baseline_name
    mkdir -p annotation_files/$baseline_name
    python make.py --baseline_path $baseline_path --baseline_name $baseline_name
    python generate.py --baseline_name $baseline_name
done
