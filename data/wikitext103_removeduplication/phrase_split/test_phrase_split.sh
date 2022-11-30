#!/bin/bash

# python test_phrase_split.py --dataset wikitext103 --recall_method bm25 --chunk_size 128
python test_phrase_split.py --dataset wikitext103 --recall_method dpr --chunk_size 128
