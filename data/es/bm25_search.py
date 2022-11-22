from es_utils import *
import pickle
from tqdm import tqdm
import argparse
import json
import ipdb


'''Generate the BM25 gray candidates:
Make sure the q-q BM25 index has been built
'''


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--pool_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--chunk_size', default=10000, type=int)
    return parser.parse_args()

def main_search(args):
    searcher = ESSearcher(f'{args["dataset"]}_phrase_copy', q_q=True)
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
    counter = 0
    chunk_prefix_path = f'../{args["dataset"]}/bm25_search_chunk'
    pbar = tqdm(total=len(keys))
    for idx in range(0, len(keys), args['batch_size']):
        document_batch = [doc for doc, index in datasets[idx:idx+args['batch_size']]]
        index_batch = keys[idx:idx+args['batch_size']]
        results = searcher.msearch(document_batch, topk=args['pool_size'])
        for d, i, r in zip(document_batch, index_batch, results):
            r = set(r) - set(i)
            collector.append((i, r))
        if len(collector) > args['chunk_size']:
            pickle.dump(collector, open(f'{chunk_prefix_path}_{counter}.pkl', 'wb'))
            counter += 1
            collector = []
        pbar.update(len(index_batch))
    if len(collector) > 0:
        pickle.dump(collector, open(f'{chunk_prefix_path}_{counter}.pkl', 'wb'))

if __name__ == '__main__':
    args = vars(parser_args())
    main_search(args)
