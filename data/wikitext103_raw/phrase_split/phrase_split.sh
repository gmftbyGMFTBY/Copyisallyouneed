#!/bin/bash

python phrase_split.py --worker_id 0 --dataset wikitext103 --recall_method dpr --chunk_size 128
python stat.py --recall_method dpr --chunk_length 128
