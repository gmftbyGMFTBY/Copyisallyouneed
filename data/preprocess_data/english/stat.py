import torch
import pickle
from tqdm import tqdm
import ipdb
import numpy as np
import json

data = []
with open('searched_results_0.txt') as f:
    for line in f.readlines():
        data.append(json.loads(line))
print(f'[!] find {len(data)} samples')

find_count, find_num, find_string_length = 0, [], []
overall_length = []
overall_count = 0 
for session in tqdm(data):
    session = session['results']
    for item in session:
        string, items = item
        if items:
            # doc = base_data[docid].strip()
            find_count += 1
            find_string_length.append(len(string.split()))
        overall_count += 1
        overall_length.append(len(string.split()))
print(f'[!] find ratio: {round(sum(find_string_length)/sum(overall_length), 4)}\naverage string length: {round(np.mean(find_string_length), 4)}\n')
print(f'[!] min length: {min(find_string_length)}; max length: {max(find_string_length)}')
