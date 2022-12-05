from datasets import load_dataset
import pprint
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
import ipdb


dataset = load_dataset('wikitext', 'wikitext-103-v1')

for mode in ['train', 'validation', 'test']:
    article_counter, document_counter = 0, 0
    data = dataset[mode]
    texts = []
    pbar = tqdm(data)
    for item in pbar:
        text = item['text'].strip()
        if text:
            if text.startswith('= '):
                # new article
                article_counter += 1
            if text.startswith('=') is False:
                document_counter += 1
                texts.append((text, f'{article_counter},{document_counter}'))
        pbar.set_description(f'[!] article id: {article_counter}; document id: {document_counter}')

    with open(f'base_data_{mode}.txt', 'w') as f:
        for line, label in tqdm(texts):
            f.write(line + '\t' + label + '\n')
