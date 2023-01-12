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
        times = []
        for item in tqdm(test_set):
            time_cost = item['time_cost']
            times.append(time_cost)
    time_cost = round(np.mean(times), 4)
    print(f'[!] average time cost: {time_cost}')
    return time_cost

if __name__ == "__main__":
    args = vars(parse_config())
    time_cost = load_result(args['test_path'])
    print('Results for', args['test_path'], 'Average Time Cost:', time_cost, file=open(f'{args["test_path"]}_result.txt', 'w'))
