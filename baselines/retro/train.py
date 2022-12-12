import torch
from tqdm import tqdm
import torch.nn as nn
import ipdb
from retro_pytorch import RETRO, TrainingWrapper
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

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
    chunk_size = 64,                              # chunk size (64 in paper)
    documents_path = './text_folder',              # path to folder of text
    glob = '**/*.txt',                             # text glob
    chunks_memmap_path = './train.chunks.dat',     # path to chunks
    seqs_memmap_path = './train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = './train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    max_chunks = 10_000_000,                        # maximum cap to chunks
    max_seqs = 2_000_000,                           # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '10G',
    current_memory_available = '50G'
)

# get the dataloader and optimizer (AdamW with all the correct settings)

train_dl = iter(wrapper.get_dataloader(batch_size = 64, shuffle = True))
optim = wrapper.get_optimizer(lr = 3e-4, wd = 0.01)

# now do your training
# ex. one gradient step

seq, retrieved = map(lambda t: t.cuda(), next(train_dl))

# seq       - (2, 2049)         - 1 extra token since split by seq[:, :-1], seq[:, 1:]
# retrieved - (2, 32, 2, 128)   - 128 since chunk + continuation, each 64 tokens

# packup the model with dataparallel
retro = nn.DataParallel(retro)
retro = retro.cuda()

# trainer
pbar = tqdm(range(100000))
save_every = 10000
for i in pbar:
    try:
        seq, retrieved = map(lambda t: t.cuda(), next(train_dl))
    except:
        # re-create the dataloader
        train_dl = iter(wrapper.get_dataloader(batch_size = 64, shuffle = True))
        seq, retrieved = map(lambda t: t.cuda(), next(train_dl))

    loss = retro(
        seq,
        retrieved,
        return_loss = True
    )
    loss = loss.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(retro.parameters(), 1.)
    optim.step()
    optim.zero_grad()
    pbar.set_description(f'[!] loss: {round(loss.item(), 4)}')

    if i % save_every == 0:
        torch.save(retro.state_dict(), f'best_model_{i}.pt')
print(f'[!] train over')
