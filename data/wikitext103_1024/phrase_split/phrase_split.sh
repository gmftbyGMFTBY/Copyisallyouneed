#!/bin/bash

worker=(0 1 2 3 4 5 6 7)
for i in ${worker[@]}
do
    echo "worker $i begin"
    python phrase_split.py --worker_id $i --dataset wikitext103 --recall_method dpr --chunk_size 128
    python stat.py --recall_method dpr --chunk_length 128
done
