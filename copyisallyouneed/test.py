from header import *
from dataloader import *
from models import *
from config import *
import sys
sys.path.append('../data/')
from dpr_1024 import Retriever

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    return parser.parse_args()

def main_generation(**args):
    retriever = Retriever('../data/wikitext103_1024/base_data_128.txt', 200, '../data/dpr_1024', 0)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model('../ckpt/wikitext103/copyisallyouneed/best_2002_10000.pt')
    print(f'[!] init model over')

    torch.manual_seed(1.0)
    torch.cuda.manual_seed_all(1.0)

    collection = []
    with open(f'../data/wikitext103_1024/test.txt') as f:
        for idx, line in enumerate(f.readlines()):
            ids = agent.model.tokenizer.encode(line, add_special_tokens=False)
            prefix, reference = ids[:32], ids[32:]
            prefix = agent.model.tokenizer.decode(prefix)
            reference = agent.model.tokenizer.decode(reference)
            text, candidates = agent.generate_one_sample(prefix, retriever, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.)
            collection.append((prefix, reference, text, candidates))
            ipdb.set_trace()
    return collection

if __name__ == "__main__":
    args = vars(parser_args())
    result = main_generation(**args)
