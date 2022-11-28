import torch
import pickle
import ipdb
from torch.utils.data import dataset, dataloader
import argparse
from tqdm import tqdm
import torch.distributed as dist
from utils import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pool_size', default=256, type=int)
    parser.add_argument('--chunk_size', default=256, type=int)
    parser.add_argument('--chunk_length', default=128, type=int)
    return parser.parse_args()

def load_base_data(path):
    datasets, keys = {}, []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            document = '\t'.join(items[:-1])
            label = items[-1].strip()
            datasets[label] = document
            keys.append(label)
    print(f'[!] load {len(datasets)} samples') 
    return datasets, keys 

def search_one_job(worker_id):
    label, embed = torch.load(f'dpr_chunk_{worker_id}_0.pt')
    print(f'[!] load {len(label)} samples from dpr_chunk_{worker_id}_0.pt')
    
    searcher = Searcher('Flat', dimension=768, nprobe=1)
    searcher.load('dpr_faiss.ckpt', 'dpr_corpus.ckpt')
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = []
    pbar = tqdm(total=len(embed))
    chunk_prefix_path = f'../{args["dataset"]}/dpr_search_chunk_{args["chunk_length"]}_{worker_id}.pkl'
    counter = 0
    for i in range(0, len(embed), args['batch_size']):
        sublabel = label[i:i+args['batch_size']]
        subembed = embed[i:i+args['batch_size']]
        result = searcher._search(subembed.numpy(), topk=args['pool_size'])
        for l, rest in zip(sublabel, result):
            collection.append((l, rest))
        pbar.update(len(sublabel))
    pickle.dump(collection, open(f'{chunk_prefix_path}', 'wb'))
    print(f'[!] save data into {chunk_prefix_path}')

if __name__ == '__main__':
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    datasets, keys = load_base_data(f'../{args["dataset"]}/base_data_{args["chunk_length"]}.txt')
    search_one_job(args['local_rank'])

