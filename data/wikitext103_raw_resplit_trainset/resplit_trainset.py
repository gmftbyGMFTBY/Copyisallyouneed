import json
import random
import ipdb
from tqdm import tqdm

random.seed(0)

with open('base_data_128.txt') as f:
    dataset = [line.strip() for line in f.readlines()]

random_index = set(random.sample(range(len(dataset)), 10000))
test_set = [dataset[i] for i in range(len(dataset)) if i in random_index]
train_set = [dataset[i] for i in range(len(dataset)) if i not in random_index]

print(f'[!] collect {len(test_set)} test set and {len(train_set)} train set')

with open('base_data_128_train.txt', 'w') as f:
    for line in tqdm(train_set):
        f.write(line + '\n')

with open('base_data_128_test.txt', 'w') as f:
    for line in tqdm(test_set):
        items = line.split('\t')
        document = '\t'.join(items[:-1])
        f.write(document + '\n')
