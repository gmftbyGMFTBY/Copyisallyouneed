from header import *
import json
from dataloader import *
from models import *
from config import *
import sys
sys.path.append('../data/')

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--decoding_method', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    return parser.parse_args()

def main_generation(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    # agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/gpt2/best_2003_10000.pt')
    agent.load_model(f'{args["root_dir"]}/ckpt/{args["dataset"]}/gpt2/best_2004_100000.pt')
    print(f'[!] init model over')

    random_seeds = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]

    collection = {i: [] for i in random_seeds}
    # with open(f'../data/en_wiki_1024/test.txt') as f:
    with open(f'../data/wikitext103_1024/test.txt') as f:
        texts = []
        for line in tqdm(f.readlines()):
            ids = agent.model.vocab.encode(line, add_special_tokens=False)
            prefix, reference = ids[:32], ids[32:]
            if len(prefix) == 32:
                prefix = agent.model.vocab.decode(prefix)
                reference = agent.model.vocab.decode(reference)
                texts.append((prefix, reference))
        print(f'[!] collect {len(texts)} valid samples which have at least 32 tokens in prefix')

        for prefix, reference in tqdm(texts):
            for seed in random_seeds:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                text, time_cost = agent.gpt2_generation(prefix, decoding_method=args['decoding_method'], top_k=0, top_p=0.95, temp=1., get_time_cost=True)
                collection[seed].append({
                    'prefix': prefix, 
                    'reference': reference, 
                    'text': text,
                    'time_cost': time_cost
                })
    return collection

if __name__ == "__main__":
    args = vars(parser_args())
    results = main_generation(**args)
    for seed in results:
        result = results[seed]
        with open(f'raw_files/random_runs/{args["dataset"]}_gpt2_result_{args["decoding_method"]}_on_wikitext103_testset_seed_{seed}.json', 'w') as f:
            json.dump(result, f, indent=4)
