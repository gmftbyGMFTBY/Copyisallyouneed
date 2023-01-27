import sys
from header import *
from dataloader import *
from models import *
from config import *
sys.path.append('../data/')
from dpr_en_wiki_1024 import Retriever

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='en_wiki', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--decoding_method', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    parser.add_argument('--split_rate', type=float, default=0.1)
    return parser.parse_args()

def main_generation(**args):
    retriever_1 = Retriever(f'../data/en_wiki_1024/base_data_128.txt', 200, f'../data/dpr_en_wiki_1024/subindex_added', 2, split_rate=0.6, nprobe=1)
    retriever_2 = Retriever(f'../data/en_wiki_1024/base_data_128.txt', 200, f'../data/dpr_en_wiki_1024/subindex_added', 2, split_rate=0.8, nprobe=1)
    tokenizer= AutoTokenizer.from_pretrained('gpt2')
    with torch.no_grad():
        with open(f'../data/{args["dataset"]}_1024/test.txt') as f:
            texts = []
            for line in tqdm(f.readlines()):
                ids = tokenizer.encode(line, add_special_tokens=False)
                prefix, reference = ids[:32], ids[32:]
                if len(prefix) == 32:
                    prefix = tokenizer.decode(prefix)
                    reference = tokenizer.decode(reference)
                    texts.append((prefix, reference))
            print(f'[!] collect {len(texts)} valid samples which have at least 32 tokens in prefix')

            average_overlap = []

            for text in texts:
                documents_1 = retriever_1.search([text], 1024)[-1]
                documents_2 = retriever_2.search([text], 1024)[-1]
                set_1, set_2 = set(documents_1), set(documents_2)
                a = set_1 & set_2
                average_overlap.append(len(a))
    print(f'[!] average overalp is', round(np.mean(average_overlap), 4))

if __name__ == "__main__":
    args = vars(parser_args())
    main_generation(**args)
