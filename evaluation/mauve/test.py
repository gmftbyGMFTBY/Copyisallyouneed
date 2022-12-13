from tqdm import tqdm
from torch.cuda.amp import autocast
import ipdb
import mauve
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer

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
    vocab = AutoTokenizer.from_pretrained('gpt2-large')
    # dataset = load_result('copyisallyouneed_result.json')
    dataset = load_result('knnlm_result.json')
    # dataset = load_result('gpt2_result.json')
    out = mauve.compute_mauve(
        p_text=[i[0] for i in dataset], 
        q_text=[i[1] for i in dataset], 
        device_id=2, 
        max_text_length=512, 
        verbose=False, 
        mauve_scaling_factor=1.0, 
    )
    print('MAUVE:', out.mauve)
