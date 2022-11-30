#!/bin/bash

worker_id=$1
python phrase_split.py --worker_id $worker_id --dataset wikitext103 --recall_method dpr --chunk_size 128
# python phrase_split.py --worker_id $worker_id --dataset wikitext103 --recall_method bm25 --chunk_size 512 
