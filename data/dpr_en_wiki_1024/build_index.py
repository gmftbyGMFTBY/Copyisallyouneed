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
                searcher.add(embed.numpy(), text)

    searcher.save(f'dpr_faiss_{split_rate}.ckpt', f'dpr_corpus_{split_rate}.ckpt')
    print(f'[!] save faiss index over')

if __name__ == "__main__":
    split_rate = 1.0
    print(f'[!] build index with {split_rate} rate')
    num = split_rate * 21000000
    # 0.1:
    build_index('IVF10000,PQ16', num)
    
    
    # build_index('IVF100000,PQ16', num)
