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


    collection = []
    # with open(f'../data/{args["dataset"]}_1024/debug_test.txt') as f:
    # with open(f'../data/wikitext103_1024/test.txt') as f:
    with open(f'../data/{args["dataset"]}_1024/test.txt') as f:
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
            text, candidates, time_cost = agent.generate_one_sample(prefix, retriever, decoding_method=args["decoding_method"], top_k=0, top_p=0.95, temp=1., get_time_cost=True)
            collection.append({
                'prefix': prefix, 
                'reference': reference, 
                'text': text, 
                'phrases': candidates,
                'time_cost': time_cost
            })
    return collection

if __name__ == "__main__":
    args = vars(parser_args())
    result = main_generation(**args)
    # with open(f'{args["dataset"]}_copyisallyouneed_result_{args["decoding_method"]}_wikitext_and_en_wiki_index_on_wikitext_testset.json', 'w') as f:
    # with open(f'{args["dataset"]}_copyisallyouneed_result_{args["decoding_method"]}_wikitext_and_en_wiki_index_on_en_wiki_testset.json', 'w') as f:
    # with open('debug_test_generation_0.95_wikitext103_phrase_topk_1024.json', 'w') as f:
    # with open('debug_test_generation_0.98_en_wiki.json', 'w') as f:
    # with open(f'{args["dataset"]}_copyisallyouneed_result_{args["decoding_method"]}_wikitext.json', 'w') as f:
    split_rate = args['split_rate']
    with open(f'raw_files/{args["dataset"]}_copyisallyouneed_result_{args["decoding_method"]}_wikitext_index_on_wikitext103_testset_{split_rate}_nprobe_10.json', 'w') as f:
        json.dump(result, f, indent=4)
