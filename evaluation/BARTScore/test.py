from bart_score import BARTScorer
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
import ipdb
import mauve
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
import argparse
from bart_score import BARTScorer


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='gpt2_result.json')
    parser.add_argument("--device", type=int)
    return parser.parse_args()

def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference']
            result = item['text']

            reference_ids = vocab.encode(reference, add_special_tokens=False)
            if len(reference_ids) <= 130:
                dataset.append((reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    args = vars(parse_config())
    batch_size = 4
    vocab = AutoTokenizer.from_pretrained('gpt2')
    dataset = load_result(args["test_path"])
    bart_scorer = BARTScorer(device=f'cuda:{args["device"]}', checkpoint='facebook/bart-large-cnn')
    with torch.no_grad():
        scores = []
        for i in tqdm(range(len(dataset))):
            reference, result = dataset[i]
            s = bart_scorer.score([result], [reference], batch_size=4)
            scores.append(s)
        s = round(np.mean(s), 4)
    print('Results for', args['test_path'], 'BARTScore:', s, 'Dataset size', len(dataset), file=open(f'{args["test_path"]}_result.txt', 'w'))
