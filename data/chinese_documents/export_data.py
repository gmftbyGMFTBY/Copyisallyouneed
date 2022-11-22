import json
import ipdb
from tqdm import tqdm
import pickle

with open('bm25_search_result_0.txt') as f:
    dataset = [json.loads(line) for line in f.readlines()]

with open('base_data.txt') as f:
    base_data = {}
    for line in tqdm(f.readlines()):
        line = line.strip()
        if line.strip():
            items = line.split('\t')
            doc = '\t'.join(items[:-1])
            index = items[-1].strip()
            if index:
                base_data[index] = doc

docs = set()
for session in dataset:
    docs |= set([item[1][0][0] for item in session['results'] if item[1]])

subdataset = {i: base_data[i] for i in docs}
pickle.dump(subdataset, open('sub_base_data.pkl', 'wb'))
