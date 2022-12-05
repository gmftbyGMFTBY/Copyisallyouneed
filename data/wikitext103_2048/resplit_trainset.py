import json
import random
import ipdb
from tqdm import tqdm

random.seed(0)

with open('base_data.txt') as f:
    dataset = [line.strip() for line in f.readlines()]

random_index = set(random.sample(range(len(dataset)), 2000))
test_set = [dataset[i] for i in range(len(dataset)) if i in random_index]

with open('new_test.txt', 'w') as f:
    for line in tqdm(test_set):
        items = line.split('\t')
        document = '\t'.join(items[:-1])
        f.write(document + '\n')
