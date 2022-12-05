import torch
import random
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

def search_one_job(worker_id):
    label, embed = torch.load(f'dpr_chunk_{worker_id}_0.pt')
    print(f'[!] load {len(label)} samples from dpr_chunk_{worker_id}_0.pt')
    
    searcher = Searcher('Flat', dimension=768, nprobe=1)
    searcher.load('dpr_faiss.ckpt', 'dpr_corpus.ckpt')
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = []

    # random_index = random.sample(range(len(embed)), 10000)
    # embed = embed[random_index]

    pbar = tqdm(total=len(embed))
    chunk_prefix_path = f'../{args["dataset"]}/dpr_search_chunk_{args["chunk_length"]}_{worker_id}.pkl'
    counter = 0
    for i in range(0, len(embed), args['batch_size']):
        sublabel = label[i:i+args['batch_size']]
        subembed = embed[i:i+args['batch_size']]
        result = searcher._search(subembed.numpy(), topk=256)

        for l, rest in zip(sublabel, result):
            doc_label = l.split(',')[0]
            rest = [item for item in rest if doc_label != item.split(',')[0]][:64]
            collection.append((l, rest))
        pbar.update(len(sublabel))
    pickle.dump(collection, open(f'{chunk_prefix_path}', 'wb'))
    print(f'[!] save data into {chunk_prefix_path}')

if __name__ == '__main__':
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    search_one_job(args['local_rank'])

