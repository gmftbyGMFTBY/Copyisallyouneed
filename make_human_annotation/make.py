import json
import random

random.seed(0)

with open('copyisallyouneed_result.json') as f:
    copyisallyouneed_result = json.load(f)
with open('neurlab_gpt2_result_nucleus_sampling.json') as f:
    gpt2_result = json.load(f)

assert len(copyisallyouneed_result) == len(gpt2_result)
results, index = [], []
labels = [random.randint(0, 1) for _ in range(len(gpt2_result))]
for a, b, label in zip(copyisallyouneed_result, gpt2_result, labels):
    assert a['prefix'] == b['prefix']
    if label == 0:
        results.append({'prefix': a['prefix'], 'method_0': a['text'], 'method_1': b['text'], 'which one is better': None})
    else:
        results.append({'prefix': a['prefix'], 'method_0': b['text'], 'method_1': a['text'], 'which one is better': None})

max_num = 100
index = random.sample(range(len(results)), max_num)
labels = [labels[i] for i in index]
results = [results[i] for i in index]
with open('human_annotation.json', 'w') as f:
    json.dump(results, f, indent=4)
with open('index.json', 'w') as f:
    json.dump(labels, f, indent=4)
