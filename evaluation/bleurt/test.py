from transformers import AutoModelForSequenceClassification, AutoTokenizer
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



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default='gpt2_result.json')
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
            result_ids = vocab.encode(result, add_special_tokens=False)
            min_length = min(len(reference_ids), len(result_ids))
            reference_ids, result_ids = reference_ids[:min_length], result_ids[:min_length]
            reference = vocab.decode(reference_ids)
            result = vocab.decode(result_ids)
            reference = prefix + ' ' + reference
            result = prefix + ' ' + result
            dataset.append((reference, result))
    print(f'[!] collect {len(dataset)} samples')
    return dataset


if __name__ == "__main__":
    args = vars(parse_config())
    vocab = AutoTokenizer.from_pretrained('gpt2-large')
    dataset = load_result(args["test_path"])

    tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
    model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
    model.eval().cuda()
    
    with torch.no_grad():
        scores = []
        for reference, result in tqdm(dataset):
            tokenize_output = tokenizer([reference], [result], return_tensors='pt')
            items = {}
            for key in tokenize_output:
                items[key] = tokenize_output[key].cuda()
            score = model(**items)[0].squeeze()
            scores.append(score.item())
        print(f'BLEURT Scores:', round(np.mean(scores), 4))
