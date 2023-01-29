from header import *
from dataloader import *
from models import *
from config import *
import sys
sys.path.append('../data/')
from dpr_en_wiki_1024 import Retriever
# from dpr_1024 import Retriever

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--decoding_method', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    parser.add_argument('--split_rate', type=float, default=0.1)
    return parser.parse_args()

def main_generation(**args):
    retriever = Retriever(f'../data/en_wiki_1024/base_data_128.txt', 200, f'../data/dpr_en_wiki_1024/subindex_added', 0, split_rate=args['split_rate'], nprobe=100)
    # retriever = Retriever(f'../data/wikitext103_1024/base_data_128.txt', 200, f'../data/dpr_1024', 0)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/best_2003_400000.pt')
    print(f'[!] init model over')

    collection = []
    with open(f'../data/{args["dataset"]}_1024/debug_test.txt') as f:
        texts = []
        for line in tqdm(f.readlines()):
            texts.append(line.strip())
        pbar = tqdm(texts)
        for prefix in pbar:
            agent.debug_generate_one_sample(prefix, retriever, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True)

if __name__ == "__main__":
    args = vars(parser_args())
    rest = main_generation(**args)
