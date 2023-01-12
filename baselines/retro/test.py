import torch
import json
from collections import OrderedDict
import time
from tqdm import tqdm
import torch.nn as nn
import ipdb
import json
from retro_pytorch import RETRO, TrainingWrapper
from retro_pytorch.training import top_p
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
# decoding_method = 'sampling'
decoding_method = 'greedy'

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len = 512,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 12,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
)

wrapper = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = './en_wiki_text_folder',              # path to folder of text
    glob = '**/*.txt',                             # text glob
    chunks_memmap_path = './en_wiki_text_folder/train.chunks.dat',     # path to chunks
    seqs_memmap_path = './en_wiki_text_folder/train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = './en_wiki_text_folder/train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    max_chunks = 10_000_000,                        # maximum cap to chunks
    max_seqs = 2_000_000,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '10G',
    current_memory_available = '50G'
)

# packup the model with dataparallel
# load the model checkpoint
model_path = 'best_model_100000.pt'
parameters = torch.load(model_path)
new_data = OrderedDict()
for key, value in parameters.items():
    key = key.replace('module.', '')
    new_data[key] = value
retro.load_state_dict(new_data)
retro = retro.cuda().eval()

max_ctx_len = 384
# 0.95 nucleus sampling
filter_thres = 0.05 if decoding_method == 'sampling' else 2
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')    #  compatible with the GPT2 vocabulary

# set the maximum sequence length for generation (32 prefix + 128 generation)
wrapper.max_seq_len = 200

collection = []
# with open(f'../../data/wikitext103_1024/test.txt') as f:
with open(f'../../data/en_wiki_1024/test.txt') as f:
# with open(f'../../data/en_wiki_1024/test.txt') as f:
    # collect the valid prefixes
    texts = []
    for line in tqdm(f.readlines()):
        ids = gpt2_tokenizer.encode(line, add_special_tokens=False)
        prefix, reference = ids[:32], ids[32:]
        if len(prefix) == 32:
            prefix = gpt2_tokenizer.decode(prefix)
            reference = gpt2_tokenizer.decode(reference)
            texts.append((prefix, reference))
    print(f'[!] collect {len(texts)} valid samples which have at least 32 tokens in prefix')
    for prefix, reference in tqdm(texts):
        prompt = torch.LongTensor(tokenizer.encode(prefix, add_special_tokens=False)).unsqueeze(0).cuda()
        prefix_len = len(tokenizer.decode(prompt[0]))
        # filter_thres larger than 1, lead to the greedy search
        bt = time.time()
        sampled = wrapper.generate(prompt, filter_fn=top_p, filter_thres = filter_thres, temperature = 1.0) # (1, <2049) terminates early if all <eos>
        time_cost = time.time() - bt
        rest = tokenizer.decode(sampled[0])
        text = rest[prefix_len:]
        collection.append({
            'prefix': prefix, 
            'reference': reference, 
            'text': text, 
            'time_cost': time_cost
        })

with open(f'lawmt_retro_result_{decoding_method}.json', 'w') as f:
    json.dump(collection, f, indent=4)
