import pickle
import torch
import multiprocessing
from itertools import chain
import spacy
import re
import torch
import subprocess
from copy import deepcopy
import time
import random
from tqdm import tqdm
import argparse
import json
import ipdb
import multiprocessing


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bsz', default=1, type=int)
    parser.add_argument('--chunk_size', default=0, type=int)
    parser.add_argument('--worker_id', default=1, type=int)
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--recall_method', default='bm25', type=str)
    return parser.parse_args()


def load_base_data(path):
    with open(path) as f:
        dataset = {}
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            id = items[-1]
            chunk = '\t'.join(items[:-1]).strip()
            dataset[id] = chunk
    return dataset

class SearchItem:

    def __init__(
        self,
        min_length,
        max_length,
        item,
        punc,
    ):
        self.text = base_data[item[0]]
        self.data, self.data_pos = self.text.split(), []
        self.self_doc_index = item[0]

        self.candidates = [i for i in list(item[1]) if i != item[0]]
        self.candidates = [self.self_doc_index] + self.candidates
        self.min_length, self.max_length = min_length, max_length
        self.pointer = 0
        self.result = []
        self.punc = punc
        self.last_rest, self.current_rest = [], []
        self.index = item[0]
        self.cache = []
    
    def move(self):
        while not self.is_end():
            if len(self.cache) > 1 and self.cache[-1] in self.punc:
                self.last_rest = self.current_rest
                self.move_back()
                self.save_once()
            elif len(self.cache) == 1 and self.cache[0] in self.punc:
                self.save_once()
            elif self.get_length() > self.max_length:
                self.current_rest = []
            elif self.min_length <= self.get_length() <= self.max_length:
                self.search_now()

            if len(self.last_rest) > 0 and len(self.current_rest) == 0:
                self.move_back()
                self.save_once()
            elif len(self.last_rest) == 0 and len(self.current_rest) > 0:
                pass
            elif len(self.last_rest) > 0 and len(self.current_rest) > 0:
                pass
            elif len(self.last_rest) == 0 and len(self.current_rest) == 0:
                if self.min_length <= self.get_length() <= self.max_length:
                    self.save_once()
            self.move_once()
        if len(self.last_rest) == 0 and len(self.current_rest) > 0:
            self.save_once()
        elif len(self.last_rest) > 0 and len(self.current_rest) > 0:
            self.save_once()
        elif len(self.last_rest) == 0 and len(self.current_rest) == 0:
            self.save_once()

    def search_now(self):
        string = ' '.join(self.cache)
        index, docid = -1, -1
        for did in self.candidates:
            doc = base_data[did]
            try:
                index = doc.index(string)
            except:
                continue
            if did == self.self_doc_index:
                doc_prefix_ = doc[:index].replace(' ', '')
                cache_prefix_ = ' '.join([item[0] for item in self.result]).replace(' ', '')
                if doc_prefix_ != cache_prefix_ and len(doc_prefix_) < len(cache_prefix_):
                    docid = did
                    break
                else:
                    docid, index = -1, -1
            else:
                if index != -1:
                    docid = did
                    break
        if docid != -1 and index != -1:
            self.save_current_rest([(docid, index)])
        else:
            self.save_current_rest([])

    def get_length(self):
        return len(self.cache)

    def save_current_rest(self, rest):
        self.last_rest = self.current_rest
        self.current_rest = rest

    def save_once(self):
        self.result.append((' '.join(self.cache), self.current_rest))
        self.cache = []
        self.last_rest, self.current_rest = [], []

    def move_once(self):
        self.cache.append(self.data[self.pointer])
        self.pointer += 1

    def is_end(self):
        return False if self.pointer < len(self.data) else True

    def move_back(self):
        self.current_rest = self.last_rest
        self.last_rest = []
        self.cache = self.cache[:-1]
        # NOTE:
        if self.pointer < len(self.data):
            self.pointer -= 1

def search_for_multiple_instance(
    min_length,
    max_length,
    jobs,
    args,
    path
):
    punc = set([',', '.', '"', "'", '?', '!', '@', '-', '<', '>', ':', ';', '/', '_', '+', '=', '~', '`', '#', '$', '%', '^', '&', '*', '(', ')', '[', ']', '{', '}'])
    results_overall = []
    with open(path, 'w') as f:
        for item in tqdm(jobs):
            searchitem = SearchItem(min_length, max_length, item, punc)
            searchitem.move()
            results_overall.append({
                'results': clean_data(searchitem.result),
                'index': searchitem.index
            })
            if len(results_overall) % 1000 == 0:
                for item in tqdm(results_overall):
                    item = json.dumps(item, ensure_ascii=False)
                    f.write(f'{item}\n')
                results_overall = []
        if len(results_overall) > 0:
            for item in tqdm(results_overall):
                item = json.dumps(item, ensure_ascii=False)
                f.write(f'{item}\n')

def clean_data(result):
    units = []
    empty_cache = []
    for unit in result:
        if unit[1]:
            if empty_cache:
                units.append((' '.join(empty_cache), []))
            units.append(unit)
            empty_cache = []
        else:
            empty_cache.append(unit[0])
    if empty_cache:
        units.append((' '.join(empty_cache), []))
    return units

def main_search(args, jobs, idx, path):
    pbar = tqdm(jobs)
    batch_size = args['bsz']
    min_length, max_length = 2, 16
    search_for_multiple_instance(min_length, max_length, jobs, args, path)

if __name__ == '__main__':
    args = vars(parser_args())
    base_data = load_base_data(f'../base_data_{args["chunk_size"]}.txt')
    idx = args['worker_id']
    jobs = pickle.load(open(f'../{args["recall_method"]}_search_chunk_{args["chunk_size"]}_{args["worker_id"]}.pkl', 'rb'))
    jobs = random.sample(jobs, 10000)
    print(f'[!] collect {len(jobs)} data samples; begin to search for {idx} woker')
    main_search(args, jobs, idx, f'../{args["recall_method"]}_search_result_{args["chunk_size"]}_{idx}.txt')
