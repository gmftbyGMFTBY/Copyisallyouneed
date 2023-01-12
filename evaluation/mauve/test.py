from tqdm import tqdm
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

            reference_ids = vocab.encode(reference, add_special_tokens=False)
            result_ids = vocab.encode(result, add_special_tokens=False)
            # min_length = min(len(reference_ids), len(result_ids))
            # reference_ids, result_ids = reference_ids[:min_length], result_ids[:min_length]
            # reference = vocab.decode(reference_ids)
            # result = vocab.decode(result_ids)
            reference = prefix + ' ' + reference
            result = prefix + ' ' + result
            if len(reference_ids) > 0:
                dataset.append((reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset

if __name__ == "__main__":
    args = vars(parse_config())
    vocab = AutoTokenizer.from_pretrained('gpt2-large')
    dataset = load_result(args["test_path"])
    out = mauve.compute_mauve(
        p_text=[i[0] for i in dataset], 
        q_text=[i[1] for i in dataset], 
        device_id=args['device'], 
        max_text_length=512, 
        verbose=False, 
        # 2 for wikitext and lawmt; 2 and 3 and 4 for en-wiki; 2 for lawmt
        mauve_scaling_factor=2.0, 
        featurize_model_name='gpt2-large',
    )
    print('Results for', args['test_path'], 'MAUVE:', out.mauve, 'Dataset size', len(dataset), file=open(f'{args["test_path"]}_result.txt', 'w'))
    # print('Results for', args['test_path'], 'MAUVE:', out.mauve, 'Dataset size', len(dataset))
