from es_utils import *
import pickle
from tqdm import tqdm
import argparse
import json
import ipdb
import nltk


'''Generate the BM25 gray candidates:
Make sure the q-q BM25 index has been built
'''


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--chunk_length', default=128, type=int)
    parser.add_argument('--pool_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--chunk_size', default=10000, type=int)
    parser.add_argument('--prefix_len', default=32, type=int)
    return parser.parse_args()

def clean_data(tokens):
    string = ' '.join(tokens)
    string = string.replace(' , ', ',')
    string = string.replace(' .', '.')
    string = string.replace(' !', '!')
    string = string.replace(' ?', '?')
    string = string.replace(' : ', ': ')
    # string = string.replace(' \'', '\'')
    return string


def test_search(args):
    # load test set
    with open(f'../{args["dataset"]}/test.txt') as f:
        datasets = [line.strip() for line in tqdm(f.readlines())]
        test_set = []
        for line in datasets:
            words = nltk.word_tokenize(line)
            if len(words) >= 32:
                # 32 is prefix_length and 128 is the reference length; refer to SimCTG
                # prefix = clean_data(words[:32])
                prefix = clean_data(words)
                reference = clean_data(words[32:128+32])
                test_set.append((prefix, reference))
    print(f'[!] collect {len(test_set)} samples from the test set')

    searcher = ESSearcher(f'{args["dataset"]}_phrase_copy_{args["chunk_length"]}', q_q=True)
    # load base_data.txt dataset
    with open(f'../{args["dataset"]}/base_data.txt') as f:
        datasets, keys = [], []
        for line in tqdm(f.readlines()):
            items = line.split('\t')
            document = '\t'.join(items[:-1])
            index = items[-1].strip()
            datasets.append((document, index))
            keys.append(index)
    collector = []
    chunk_path = f'../{args["dataset"]}/test_bm25_search_{args["chunk_length"]}.pkl'
    print(f'[!] read data from {chunk_path}')
    pbar = tqdm(total=len(test_set))
    for idx in range(0, len(test_set), args['batch_size']):
        prefix_batch = [prefix for prefix, reference in test_set[idx:idx+args['batch_size']]]
        reference_batch = [reference for prefix, reference in test_set[idx:idx+args['batch_size']]]
        results = searcher.msearch(prefix_batch, topk=args['pool_size'])
        for p, r, re in zip(prefix_batch, reference_batch, results):
            collector.append((p, r, re))
        pbar.update(len(prefix_batch))
    pickle.dump(collector, open(chunk_path, 'wb'))

if __name__ == '__main__':
    args = vars(parser_args())
    test_search(args)
