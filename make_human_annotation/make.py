import json
import ipdb
import random
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_path', type=str)
parser.add_argument('--baseline_name', type=str)
args = vars(parser.parse_args())

# set the random seed for partial reproduction
random.seed(0)

# with open('raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_v2.json') as f:
# with open('raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_and_en_wiki_index_on_en_wiki_testset.json') as f:
# with open('raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_and_en_wiki_index_on_wikitext_testset.json') as f:
# with open('raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_on_en_wiki_testset_0.8.json') as f:
with open('raw_files/en_wiki_copyisallyouneed_result_nucleus_sampling_wikitext_index_on_wikitext103_testset_1.0_nprobe_10.json') as f:
    copyisallyouneed_result = json.load(f)
with open(args['baseline_path']) as f:
    gpt2_result = json.load(f)

assert len(copyisallyouneed_result) == len(gpt2_result)
results, index = [], []
labels = [random.randint(0, 1) for _ in range(len(gpt2_result))]

# valid_indexes = []
counter = 0
for a, b, label in zip(copyisallyouneed_result, gpt2_result, labels):
    assert a['prefix'] == b['prefix']

    # if '<|endoftext|>' in a['prefix'] or '<|endoftext|>' in a['text'] or '<|endoftext|>' in b['text']:
    #     pass
    # else:
    #     valid_indexes.append(counter)
    if label == 0:
        results.append({'prefix': a['prefix'], 'method_0': a['text'], 'method_1': b['text'], 'which one is better': None})
    else:
        results.append({'prefix': a['prefix'], 'method_0': b['text'], 'method_1': a['text'], 'which one is better': None})
    counter += 1

# print(f'[!] find {len(valid_indexes)} that doesn"t have the UNK token')

max_num = 100
# index = random.sample(range(len(valid_indexes)), max_num)
index = random.sample(range(len(labels)), max_num)
# labels = [labels[valid_indexes[i]] for i in index]
# results = [results[valid_indexes[i]] for i in index]
labels = [labels[i] for i in index]
results = [results[i] for i in index]
with open(f'annotation_files/{args["baseline_name"]}/human_annotation.json', 'w') as f:
    json.dump(results, f, indent=4)
with open(f'annotation_files/{args["baseline_name"]}/index.json', 'w') as f:
    json.dump(labels, f, indent=4)
