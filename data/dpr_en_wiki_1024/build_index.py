from utils import *
import ipdb
import torch
from tqdm import tqdm
import joblib
import numpy as np

def build_index(index_type, max_num):
    embds, texts = [], []
    current_num = 0
    added = []
    for i in tqdm(range(8)):
        for idx in range(100):
            try:
                text, embed = torch.load(
                    f'dpr_chunk_{i}_{idx}.pt'
                )
                print(f'[!] load dpr_chunk_{i}_{idx}.pt')
                current_num += len(embed)
            except Exception as error:
                print(error)
                break
            added.append((i, idx))
            embds.append(embed.numpy())
            texts.extend(text)
            print(f'[!] collect embeddings: {current_num}')
            if len(text) > max_num:
                break
        if len(texts) >= max_num:
            break
    embds = np.concatenate(embds) 
    if len(texts) > max_num:
        texts = texts[:max_num]
        embds = embds[:max_num]

    searcher = Searcher(index_type, dimension=768)
    searcher._build(embds, texts, speedup=True)
    print(f'[!] train the searcher over')
    searcher.move_to_cpu()

    if len(texts) < max_num:
        for i in tqdm(range(8)):
            for idx in range(100):
                if (i, idx) in added:
                    continue
                try:
                    text, embed = torch.load(
                        f'dpr_chunk_{i}_{idx}.pt'
                    )
                    print(f'[!] load dpr_chunk_{i}_{idx}.pt')
                    current_num += len(embed)
                except Exception as error:
                    print(error)
                    break
                delta_num = max_num - searcher.searcher.ntotal
                text = text[:delta_num]
                embed = embed.numpy()[:delta_num]
                searcher.add(embed, text)

    searcher.save(f'subindex/dpr_faiss_{split_rate}.ckpt', f'subindex/dpr_corpus_{split_rate}.ckpt')
    print(f'[!] save faiss index over')

if __name__ == "__main__":
    split_rate = 0.03
    print(f'[!] build index with {split_rate} rate')
    num = int(split_rate * 21000000)
    # en-wiki subindex
    build_index('IVF10000,PQ16', num)
    
    # build_index('IVF100000,PQ16', num)
