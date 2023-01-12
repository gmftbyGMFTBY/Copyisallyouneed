from tqdm import tqdm
from evaluate import load
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
        for item in tqdm(test_set):
            prefix = item['prefix']
            reference = item['reference'].strip()
            result = item['text']

            result = prefix + reference
            # result = prefix + result
            dataset.append(result)
    print(f'[!] collect {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    args = vars(parse_config())
    dataset = load_result(args['test_path'])
    ppl = load('perplexity', module_type='metric')
    results = ppl.compute(
        predictions=dataset, 
        model_id='gpt2-large', 
        batch_size=8, 
        add_start_token=False, 
        device='gpu',
    )
    print(args['test_path'], results['mean_perplexity'])
