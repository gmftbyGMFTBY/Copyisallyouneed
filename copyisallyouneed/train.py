from header import *
from dataloader import *
from models import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--total_workers', type=int)
    return parser.parse_args()


def main(**args):
    torch.cuda.empty_cache()
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    args['global_rank'] = dist.get_rank()
    print(f'[!] global rank: {args["global_rank"]}')

    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    train_data, train_iter, sampler = load_dataset(args)
    
    # set seed
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    if args['local_rank'] == 0:
        sum_writer = SummaryWriter(
            log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/{args["version"]}',
        )
    else:
        sum_writer = None
        
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])
    agent = load_model(args)
    pbar = tqdm(total=args['total_step'])
    current_step, over_train_flag = 0, False
    sampler.set_epoch(0)    # shuffle for DDP
    if agent.load_last_step:
        current_step = agent.load_last_step + 1
        print(f'[!] load latest step: {current_step}')
    for _ in range(100000000):
        for batch in train_iter:
            agent.train_model(
                batch, 
                recoder=sum_writer, 
                current_step=current_step, 
                pbar=pbar
            )
            if args['global_rank'] == 0 and current_step % args['save_every'] == 0 and current_step > 0:
                save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{args["version"]}_{current_step}.pt'
                agent.save_model_long(save_path, current_step)
            current_step += 1
            if current_step > args['total_step']:
                over_train_flag = True
                break
        if over_train_flag:
            break
    if sum_writer:
        sum_writer.close()

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
