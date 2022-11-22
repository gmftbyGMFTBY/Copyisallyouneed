from es_utils import *
import argparse
import random
import ipdb

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='wikitext103', type=str)
    return parser.parse_args()


def copyisallyouneed_dataset(args):
    path = f'../{args["dataset"]}/base_data.txt'
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            items = line.strip().split('\t')
            context = '\t'.join(items[:-1])
            index = items[-1].strip()
            dataset.append((context, index))
    print(f'[!] collect {len(dataset)} sampels for BM25 retrieval')
    return dataset

if __name__ == "__main__":
    args = vars(parser_args())
    data = copyisallyouneed_dataset(args)
    builder = ESBuilder(
        f'{args["dataset"]}_phrase_copy',
        create_index=True,
        q_q=True,
    )
    builder.insert(data)
