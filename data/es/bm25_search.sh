#!/bin/bash

# python bm25_search.py --dataset chinese_documents --pool_size 64 --batch_size 64
python bm25_search.py --dataset wikitext103 --pool_size 64 --batch_size 64 --chunk_length 128 --chunk_size 1000000
