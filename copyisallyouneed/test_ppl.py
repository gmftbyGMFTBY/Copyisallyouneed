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
    return parser.parse_args()

def test_ppl(**args):
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    # agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/gpt2/best_2003_10000.pt')
    print(f'[!] init model over')

    test_data, test_iter, sampler = load_dataset(args)
    agent.test_model_ppl(test_iter)

if __name__ == "__main__":
    args = vars(parser_args())
    test_ppl(**args)
