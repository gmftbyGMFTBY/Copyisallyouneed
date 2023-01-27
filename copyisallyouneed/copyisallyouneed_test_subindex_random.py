from header import *
from dataloader import *
from models import *
from config import *
import sys
sys.path.append('../data/')
from dpr_en_wiki_1024 import Retriever

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--decoding_method', type=str)
    parser.add_argument('--recall_topk', type=int, default=20)
    parser.add_argument('--split_rate', type=float, default=0.1)
    return parser.parse_args()

def main_generation(**args):
    retriever = Retriever(f'../data/en_wiki_1024/base_data_128.txt', 200, f'../data/dpr_en_wiki_1024/subindex_added', 0, split_rate=args['split_rate'], nprobe=10)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/copyisallyouneed/best_2003_400000.pt')
    print(f'[!] init model over')
    # random_seeds = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]
    random_seeds = [1.0]

    collection = {i: [] for i in random_seeds}
    # with open(f'../data/{args["dataset"]}_1024/test.txt') as f:
    with open(f'../data/wikitext103_1024/test.txt') as f:
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
            # multiple_samples = agent.generate_multiple_sample(prefix, retriever, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True, random_seeds=random_seeds, reference=reference)
            samples = agent.generate_one_sample(prefix, retriever, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True)
            for key in multiple_samples:
                collection[key].append(multiple_samples[key])
    return collection

if __name__ == "__main__":
    args = vars(parser_args())
    results = main_generation(**args)
    split_rate = args['split_rate']
    for seed in results:
        result = results[seed]
        with open(f'raw_files/random_runs_en_wiki_testset/{args["dataset"]}_copyisallyouneed_result_{args["decoding_method"]}_on_wikitext103_testset_seed_{seed}_{split_rate}.json', 'w') as f:
            json.dump(result, f, indent=4)
