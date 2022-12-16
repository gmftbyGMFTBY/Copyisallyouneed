from tqdm import tqdm
import json
import spacy
import jieba
import ipdb
import numpy as np
from transformers import AutoTokenizer

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

def distinct_sentence_level_char(sentence, n=1):
    items = [token.text for token in nlp(sentence)]
    base = []
    for i in range(0, len(items) - n + 1):
        base.append(tuple(items[i:i+n]))
    try:
        return 1 - len(set(base)) / len(base)
    except:
        return 1.

if __name__ == "__main__":
    vocab = AutoTokenizer.from_pretrained('gpt2-large')
    # dataset = load_result('copyisallyouneed_result.json')
    # dataset = load_result('knnlm_result.json')
    # dataset = load_result('gpt2_result.json')
    # dataset = load_result('gpt2_result_greedy.json')
    # dataset = load_result('copyisallyouneed_result_greedy.json')
    # dataset = load_result('retro_result_greedy.json')
    # dataset = load_result('retro_result_sampling.json')
    dataset = load_result('neurlab_gpt2_result_greedy.json')
    # dataset = load_result('neurlab_gpt2_result_nucleus_sampling.json')
    n = [2, 3, 4]
    nlp = spacy.load('en_core_web_sm')
    func = distinct_sentence_level_char
    scores = []
    for _, ca in tqdm(dataset):
        cache = {}
        for n_ in n:
            cache[f'rep-{n_}'] = func(ca, n=n_)
        scores.append(cache)

    diversity_score = 1
    for n_ in n:
        s = 1 - np.mean([item[f'rep-{n_}'] for item in scores])
        s_ = np.mean([item[f'rep-{n_}'] for item in scores])
        diversity_score *= s
        print(f'rep-{n_}', round(s_, 4))
    print(f'[!] diversity(2-4): {round(diversity_score, 4)}')
