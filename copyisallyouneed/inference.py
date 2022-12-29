from header import *
from dataloader import *
from models import *
from config import *

'''only for KNN-LM Baseline'''

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parser_args())
    # init the model
    args['mode'] = 'inference'
    
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    agent.load_model(f'{args["root_dir"]}/ckpt/wikitext103/gpt2/best_{args["version"]}_10000.pt')
    print(f'[!] load the KNN-LM model over')
    

    # inference 
    args['global_rank'] = dist.get_rank()
    data, data_iter, sampler = load_dataset(args)
    sampler.set_epoch(0)
    agent.inference_knnlm(data_iter)

    # barries
    torch.distributed.barrier()
