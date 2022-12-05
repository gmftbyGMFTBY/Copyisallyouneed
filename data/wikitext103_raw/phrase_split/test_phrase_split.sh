#!/bin/bash

python test_phrase_split.py --dataset wikitext103 --recall_method dpr --chunk_size 128
python stat_test.py --recall_method dpr --chunk_length 128
