from utils import *
import ipdb
import torch
from tqdm import tqdm
import joblib
import numpy as np

def build_index(index_type):
    embds, texts = [], []
    current_num = 0
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
            embds.append(embed.numpy())
            texts.extend(text)
            print(f'[!] collect embeddings: {current_num}')
    embds = np.concatenate(embds) 
    searcher = Searcher(index_type, dimension=768)
    searcher._build(embds, texts, speedup=True)
    print(f'[!] train the searcher over')
    searcher.move_to_cpu()

    searcher.save('dpr_faiss.ckpt', 'dpr_corpus.ckpt')
    print(f'[!] save faiss index over')

if __name__ == "__main__":
    build_index('Flat')
