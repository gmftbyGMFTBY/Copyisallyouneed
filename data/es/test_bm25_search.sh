#!/bin/bash

# python bm25_search.py --dataset chinese_documents --pool_size 64 --batch_size 64
python test_bm25_search.py --dataset wikitext103 --pool_size 256 --batch_size 64 --prefix_len 32 --chunk_length 128
