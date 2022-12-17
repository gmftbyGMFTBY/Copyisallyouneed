import json
import re
import ipdb
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

def sentence_token_nltk(str):
    sent_tokenize_list = re.split(r'([.,])', str)
    sents = []
    for item in sent_tokenize_list:
        if item in [',', '.']:
            sents[-1] += item
        else:
            sents.append(item)
    return sents

chunk_size = 128

with open('base_data.txt') as f:
    datasets = ['\t'.join(line.strip().split('\t')[:-1]) for line in f.readlines()]
    new_datasets, idx = [], 0
    for item in tqdm(datasets):
        item = item.replace(' @-@ ', '-').replace(' @,@ ', ', ').replace(' @.@ ', '.').replace('<unk>', '<|endoftext|>').strip()
        sentences = sentence_token_nltk(item)
        cache, counter = [], 0
        for sent in sentences:
            tokens = sent.split()
            if len(cache) + len(tokens) > chunk_size and len(cache) > 0:
                new_datasets.append((' '.join(cache), f'{idx},{counter}'))
                cache = tokens
                counter += 1
            else:
                cache.extend(tokens)
        if len(cache) > 0:
            new_datasets.append((' '.join(cache), f'{idx},{counter}'))
        idx += 1

print(f'[!] collect {len(new_datasets)} chunks from the base_data.txt')

with open(f'base_data_{chunk_size}.txt', 'w') as f:
    for chunk, label in tqdm(new_datasets):
        string = f'{chunk}\t{label}\n'
        f.write(string)
