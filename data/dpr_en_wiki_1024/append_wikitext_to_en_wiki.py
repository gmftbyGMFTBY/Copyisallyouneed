import faiss
import torch
from utils import *
from tqdm import tqdm
import numpy as np

split_rate = 0.03

en_wiki_searcher = Searcher('Flat', dimension=768, nprobe=1)
en_wiki_searcher.load(f'subindex/dpr_faiss_{split_rate}.ckpt', f'subindex/dpr_corpus_{split_rate}.ckpt')

embds, texts = [], []
current_num = 0
for i in tqdm(range(8)):
    for idx in range(100):
        try:
            text, embed = torch.load(
                f'../dpr_1024/previous_dpr_chunks/dpr_chunk_{i}_{idx}.pt'
            )
            print(f'[!] load dpr_chunk_{i}_{idx}.pt')
            current_num += len(embed)
        except Exception as error:
            print(error)
            break
        embds.append(embed.numpy())
        text = ['wikitext,' + i for i in text]
        texts.extend(text)
        print(f'[!] collect embeddings: {current_num}')
embds = np.concatenate(embds) 
en_wiki_searcher.add(embds, texts)
print(f'[!] add the wikitext-103 index over')
en_wiki_searcher.save(f'subindex_added/dpr_faiss_{split_rate}.ckpt', f'subindex_added/dpr_corpus_{split_rate}.ckpt')
print(f'[!] save faiss index over')


