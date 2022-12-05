import torch
import nltk
import pickle
import ipdb
from torch.utils.data import dataset, dataloader
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
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
            # if label.endswith(',0'):
            datasets[label] = document
            keys.append(label)
    print(f'[!] load {len(datasets)} samples') 
    return datasets, keys 

def clean_data(tokens):
    string = ' '.join(tokens)
    string = string.replace(' , ', ',')
    string = string.replace(' .', '.')
    string = string.replace(' !', '!')
    string = string.replace(' ?', '?')
    string = string.replace(' : ', ': ')
    # string = string.replace(' \'', '\'')
    return string

def search_one_job(worker_id):

    # encode the test prefix
    with open(f'../{args["dataset"]}/test.txt') as f:
        datasets = [line.strip() for line in tqdm(f.readlines())]
        test_set = []
        for line in datasets:
            words = nltk.word_tokenize(line)
            if len(words) >= 32:
                prefix = clean_data(words[:32])
                # prefix = clean_data(words)
                reference = clean_data(words[32:32+128])
                test_set.append((prefix, reference))
    print(f'[!] collect {len(test_set)} samples from the test set')

    ## encode
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
    with torch.no_grad():
        embeds = []
        for idx in tqdm(range(0, len(test_set), 256)):
            test_list = [prefix for prefix, reference in test_set[idx:idx+256]]
            batch = tokenizer.batch_encode_plus(test_list, padding=True, return_tensors='pt', max_length=256, truncation=True)
            input_ids = batch['input_ids'].cuda()
            mask = batch['attention_mask'].cuda()
            embeddings = model(input_ids=input_ids, attention_mask=mask).pooler_output    # [B, E]
            embeddings = embeddings.cpu() 
            embeds.append(embeddings)
        embeds = torch.cat(embeds).numpy()
        assert len(embeds) == len(test_set)
    
    searcher = Searcher('Flat', dimension=768, nprobe=1)
    searcher.load('dpr_faiss.ckpt', 'dpr_corpus.ckpt')
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = []
    pbar = tqdm(total=len(test_set))
    chunk_prefix_path = f'../{args["dataset"]}/test_dpr_search_{args["chunk_length"]}.pkl'
    counter = 0
    for i in range(0, len(test_set), args['batch_size']):
        subprefix = [prefix for prefix, reference in test_set[i:i+args['batch_size']]]
        subreference = [reference for prefix, reference in test_set[i:i+args['batch_size']]]
        subembed = embeds[i:i+args['batch_size']]
        result = searcher._search(subembed, topk=args['pool_size'])
        for p, r, rest in zip(subprefix, subreference, result):
            collection.append((p, r, rest))
        pbar.update(len(subreference))
    pickle.dump(collection, open(f'{chunk_prefix_path}', 'wb'))
    print(f'[!] save data into {chunk_prefix_path}')

if __name__ == '__main__':
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    # datasets, keys = load_base_data(f'../{args["dataset"]}/base_data_{args["chunk_length"]}.txt')
    search_one_job(args['local_rank'])

