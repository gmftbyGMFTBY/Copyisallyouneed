import torch
import random
import pickle
from tqdm import tqdm
import ipdb
import numpy as np
import json
import argparse
import nltk

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker_id', default=0, type=int)
    parser.add_argument('--chunk_length', default=128, type=int)
    parser.add_argument('--recall_method', default='bm25', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = vars(parser_args())
    data = []
    with open(f'../{args["recall_method"]}_search_result_{args["chunk_length"]}_{args["worker_id"]}.txt') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    print(f'[!] find {len(data)} samples')

    find_count, notfind_count, find_ngram, notfind_ngram= 0, 0, {}, {}
    overall_length = []
    overall_count = 0 
    for session in tqdm(data):
        session = session['results']
        for phrase, metadata in session:
            words = nltk.word_tokenize(phrase)
            if metadata:
                doc, doc_pos = metadata[0]
                find_count += len(words)
                if len(words) not in find_ngram:
                    find_ngram[len(words)] = 1
                else:
                    find_ngram[len(words)] += 1
            else:
                if len(words) not in notfind_ngram:
                    notfind_ngram[len(words)] = 1
                else:
                    notfind_ngram[len(words)] += 1
                notfind_count += len(words) 
    print(f'[!] find ratio: {round(find_count/(find_count+notfind_count), 4)}\n')
    sum_count = sum([value for key, value in find_ngram.items() if 0 < key <=7])
    # find ngram
    for key, value in sorted(find_ngram.items(), key=lambda x: x[0]):
        if 0 < key <= 7:
            print(f'[!] found {key}-gram: {round(value/sum_count, 4)}')
