import json
import ipdb
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import spacy

def sentence_token_nltk(str):
    sent_tokenize_list = sent_tokenize(str)
    return sent_tokenize_list

chunk_size = 128
nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'textcat', 'parser', 'tok2vec', 'lemmatizer', 'attribute_ruler'])

with open('base_data_128.txt') as f:
# with open('test.txt') as f:
    new_datasets, idx = [], 0
    for item in tqdm(f.readlines()):
        items = item.split('\t')
        string = '\t'.join(items[:-1])
        string = ' '.join([token.text for token in nlp(string)])
        string += '\t' + items[-1]

        # string = ' '.join([token.text for token in nlp(item)])
        new_datasets.append(string)

print(f'[!] collect {len(new_datasets)} chunks from the base_data.txt')

# with open(f'test_tokenized.txt', 'w') as f:
with open(f'base_data_128_tokenized.txt', 'w') as f:
    for line in tqdm(new_datasets):
        f.write(line)
