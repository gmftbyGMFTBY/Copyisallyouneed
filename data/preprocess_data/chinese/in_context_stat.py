import torch
import pickle
from tqdm import tqdm
import ipdb
import numpy as np
import json

data = []
with open('searched_results_0.txt') as f:
    counter = []
    file_counter = 0
    size = 0
    for line in tqdm(f.readlines()):
        line = json.loads(line)
        index = line['index']
        result = line['results']
        flag = False
        for phrase, docs in result:
            if docs:
                doc = docs[0][0]
                if doc == index:
                    counter.append(1)
                    flag = True
                else:
                    counter.append(0)
        if flag:
            file_counter += 1
        size += 1
    print(f'[!] rate: {np.mean(counter)}')
    print(f'[!] file rate: {file_counter/size}')
