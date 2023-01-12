from tqdm import tqdm
import numpy as np
from torch.cuda.amp import autocast
import ipdb
import mauve
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
import argparse

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='gpt2_result.json')
    parser.add_argument("--device", type=int)
    return parser.parse_args()

def load_result(path):
    with open(path) as f:
        test_set = json.load(f)
        dataset = []
        ngrams = {i: 0 for i in range(1, 100)}
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference'].strip()
            phrases = item['phrases']
            copied = []
            for phrase in phrases:
                if type(phrase) == int:
                    ngrams[1] += 1
                else:
                    copied.append(phrase)
            try:
                phrases = vocab.batch_encode_plus(copied, add_special_tokens=False)['input_ids']
            except:
                continue
            for i in phrases:
                ngrams[len(i)] += 1

    sum_num = sum([ngrams[k] for k in ngrams if k <= 6])
    for i in ngrams:
        if i <= 6:
            print(f'[!] {i}-gram: {round(ngrams[i]/sum_num, 4)}')

    list_length = []
    for i in ngrams:
        if i <= 6:
            list_length.extend([i] * ngrams[i])
    mean_length = np.mean(list_length)
    print(f'[!] mean phrase length: {round(mean_length, 4)}; variance: {round(np.var(list_length), 4)}')

if __name__ == "__main__":
    args = vars(parse_config())
    vocab = AutoTokenizer.from_pretrained('gpt2-large')
    load_result(args["test_path"])
