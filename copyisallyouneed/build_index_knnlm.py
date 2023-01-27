import ipdb
import torch
from tqdm import tqdm
import joblib
import numpy as np
import sys
sys.path.append('../data')
from dpr_1024 import *

'''build the KNN-LM Index'''

def build_index(index_type, root_dir):
    embds, texts = [], []
    current_num = 0
    for idx in range(210):
        try:
            embed, text = torch.load(f'{root_dir}/inference_0_{idx}.pt')
            print(f'[!] load {root_dir}/inference_0_{idx}.pt')
            current_num += len(embed)
        except Exception as error:
            print(error)
            break
        embds.append(embed)
        texts.extend(text)
        print(f'[!] collect embeddings: {current_num}')
    embds = np.concatenate(embds) 
    searcher = Searcher(index_type, dimension=768)
    searcher._build(embds, texts, speedup=False)
    print(f'[!] train the searcher over')
    searcher.save(f'{root_dir}/knnlm_faiss.ckpt', f'{root_dir}/knnlm_corpus.ckpt')
    print(f'[!] save faiss index over')

if __name__ == "__main__":
    # IVF100000,PQ16 for wikitext103 
    # build_index('IVF100000,PQ16', f'/apdcephfs/share_916081/johntianlan/copyisallyouneed/data/wikitext103_1024/knnlm')
    
    # IVF100000,PQ16 for lawmt
    build_index('IVF10000,PQ16', f'/apdcephfs/share_916081/johntianlan/copyisallyouneed/data/lawmt_1024/knnlm')

    # IVF100000,PQ16 for en-wiki (3B tokens)
    # build_index('IVF2000000,PQ16', f'/apdcephfs/share_916081/johntianlan/copyisallyouneed/data/{args["dataset"]}_1024/knnlm')
