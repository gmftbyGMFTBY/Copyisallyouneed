import torch
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
import ipdb
from retro_pytorch import RETRO, TrainingWrapper
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len = 256,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
)

wrapper = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = './text_folder',              # path to folder of text
    glob = '**/*.txt',                             # text glob
    chunks_memmap_path = './train.chunks.dat',     # path to chunks
    seqs_memmap_path = './train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    max_chunks = 10_000_000,                        # maximum cap to chunks
    max_seqs = 100_000,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '10G',
    current_memory_available = '50G'
)

# get the dataloader and optimizer (AdamW with all the correct settings)

train_dl = iter(wrapper.get_dataloader(batch_size = 32, shuffle = True))
optim = wrapper.get_optimizer(lr = 3e-4, wd = 0.01)

# now do your training
# ex. one gradient step

seq, retrieved = map(lambda t: t.cuda(), next(train_dl))

# seq       - (2, 2049)         - 1 extra token since split by seq[:, :-1], seq[:, 1:]
# retrieved - (2, 32, 2, 128)   - 128 since chunk + continuation, each 64 tokens

# packup the model with dataparallel
parameters = torch.load('best_model.pt')

new_data = OrderedDict()
for key, value in parameters.items():
    key = key.replace('module.', '')
    new_data[key] = value

retro.load_state_dict(new_data)
retro = retro.cuda()
retro.eval()

# topk sampling with retrieval at chunk boundaries
# sampled = wrapper.generate(filter_thres = 0.9, temperature = 1.0) # (1, <2049) terminates early if all <eos>

# or you can generate with a prompt, knn retrieval for initial chunks all taken care of

# prompt = torch.randint(0, 1000, (1, 128))  # start with two chunks worth of sequence
prefix = 'Robert Boulter is'
prompt = torch.LongTensor(tokenizer.encode(prefix, add_special_tokens=False)).unsqueeze(0)
sampled = wrapper.generate(prompt, filter_thres = 0.9, temperature = 1.0) # (1, <2049) terminates early if all <eos>
