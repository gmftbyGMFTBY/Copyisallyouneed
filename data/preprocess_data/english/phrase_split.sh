#!/bin/bash

worker_id=$1
python phrase_split.py --worker_id $worker_id --dataset chinese_documents
