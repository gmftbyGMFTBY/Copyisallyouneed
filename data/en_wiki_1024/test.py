import numpy as np
from tqdm import tqdm
import ipdb
from transformers import AutoTokenizer

f1 = open('base_data_128.txt')
f2 = open('base_data_128_original.txt')

error_num =0 

for i in tqdm(range(100)):
    l1 = f1.readline()
    l2 = f2.readline()
    vocab = AutoTokenizer.from_pretrained('gpt2')
    a1 = vocab.encode(l1)
    b1 = vocab.encode(l2)

    try:
        ipdb.set_trace()
        assert a1 == b1
    except:
        error_num += 1

    print(f'[!] error rate: {error_num/1000}')
