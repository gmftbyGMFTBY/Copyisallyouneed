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
    parser.add_argument('--decoding_method', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    return parser.parse_args()

def main_generation(**args):
    retriever = Retriever('../data/wikitext103_1024/base_data_128.txt', 200, '../data/dpr_1024', 0)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/best_2003_400000.pt')
    print(f'[!] init model over')

    torch.manual_seed(1.0)
    torch.cuda.manual_seed_all(1.0)

    collection = []
    with open(f'../data/wikitext103_1024/test.txt') as f:
        # collect the valid prefixes
        texts = []
        for line in tqdm(f.readlines()):
            ids = agent.model.tokenizer.encode(line, add_special_tokens=False)
            prefix, reference = ids[:32], ids[32:]
            if len(prefix) == 32:
                prefix = agent.model.tokenizer.decode(prefix)
                reference = agent.model.tokenizer.decode(reference)
                texts.append((prefix, reference))
        print(f'[!] collect {len(texts)} valid samples which have at least 32 tokens in prefix')
        for prefix, reference in tqdm(texts):
            text, candidates = agent.generate_one_sample(prefix, retriever, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1.)
            collection.append({
                'prefix': prefix, 
                'reference': reference, 
                'text': text, 
                'phrases': candidates
            })
            ipdb.set_trace()
    return collection

if __name__ == "__main__":
    args = vars(parser_args())
    result = main_generation(**args)
    with open(f'copyisallyouneed_result_{args["decoding_method"]}.json', 'w') as f:
        json.dump(result, f, indent=4)
