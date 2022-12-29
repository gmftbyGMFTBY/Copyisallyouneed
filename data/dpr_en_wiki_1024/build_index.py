from utils import *
import ipdb
import torch
from tqdm import tqdm
import joblib
import numpy as np

def build_index(index_type):
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
        if len(texts) >= 7000000:
            break
    
    embds = np.concatenate(embds) 
    searcher = Searcher(index_type, dimension=768)
    searcher._build(embds, texts, speedup=False)
    print(f'[!] train the searcher over')
    searcher.move_to_cpu()

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

    searcher.save('dpr_faiss.ckpt', 'dpr_corpus.ckpt')
    print(f'[!] save faiss index over')

if __name__ == "__main__":
    build_index('IVF100000,PQ16')
