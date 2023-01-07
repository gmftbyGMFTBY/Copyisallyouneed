import json
import os
import ipdb
from tqdm import tqdm
from transformers import AutoTokenizer
import pickle
import random
import ipdb
import xlrd
import xlwt
import argparse


def load_evaluation_file(file_name):

    wb = xlrd.open_workbook(file_name)
    sheet = wb.sheet_by_name('标注数据')
    nrows = sheet.nrows
    dataset = []
    counter = 1
    for i in range(nrows):
        row = sheet.row_values(i)
        if f'前缀-{counter}' == row[0]:
            dataset.append([row])
            counter += 1
        else:
            dataset[-1].append(row)
    print(f'[!] collect {len(dataset)} samples')
    
    results = []
    for sample  in dataset:
        a, b, c = sample[0], sample[1], sample[2]
        if b[-1] == c[-1]:
            results.append(2)
        elif b[-1] == 1 and c[-1] == '':
            results.append(0)
        elif b[-1] == '' and c[-1] == 1:
            results.append(1)
        else:
            ipdb.set_trace()
    return results

# root_dir = ['gpt2_nucleus_sampling', 'neurlab_gpt2_nucleus_sampling', 'retro_nucleus_sampling', 'knnlm_nucleus_sampling']
root_dir = ['en_wiki_gpt2_nucleus_sampling']
# root_dir = ['lawmt_gpt2_nucleus_sampling']

dataset = [
    load_evaluation_file(f'{root_dir[0]}/copyisallyouneed-gpt2_nucleus_sampling.xls'),
    # load_evaluation_file(f'{root_dir[1]}/copyisallyouneed-neurlab_gpt2_nucleus_sampling.xls'),
    # load_evaluation_file(f'{root_dir[2]}/copyisallyouneed-retro_nucleus_sampling.xls'),
    # load_evaluation_file(f'{root_dir[3]}/copyisallyouneed-knnlm_nucleus_sampling.xls')
]
index = json.load(open(f'{root_dir[0]}/index.json'))

for name, data in zip(root_dir, dataset):
    win_num, loss_num, tie_num = 0, 0, 0
    for i, l in zip(data, index):
        if i == 2:
            tie_num += 1
            continue
        if i == l:
            win_num += 1
        else:
            loss_num += 1

    print(f'[!] Results of {name}')
    print(f'[!] Win num: {win_num}')
    print(f'[!] Loss num: {loss_num}')
    print(f'[!] Tie num: {tie_num}\n')

