from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import ipdb
import argparse
from tqdm import tqdm
import torch.distributed as dist

def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--data_path', default='ecommerce', type=str)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--cut_size', type=int, default=500000)
    return parser.parse_args()


class DPRDataset(Dataset):

    def __init__(self, path):
        self.data = []
        with open(path) as f:
            for line in tqdm(f.readlines()):
                items = line.strip().split('\t')
                document = '\t'.join(items[:-1])
                label = items[-1]
                self.data.append((document, label))
        print(f'[!] load {len(self.data)} samples') 
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate(self, batch):
        document = [i[0] for i in batch]
        label = [i[1] for i in batch]
        return document, label

def inference_one_batch(text_list):
    with torch.no_grad():
        batch = tokenizer.batch_encode_plus(text_list, padding=True, return_tensors='pt', max_length=256, truncation=True)
        input_ids = batch['input_ids'].cuda()
        mask = batch['attention_mask'].cuda()
        embeddings = model(input_ids=input_ids, attention_mask=mask).pooler_output
    return embeddings.cpu() 

def inference(**args):
    data = DPRDataset(args['data_path'])
    sampler = torch.utils.data.distributed.DistributedSampler(data)
    data_iter = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    sampler.set_epoch(0)

    text_lists, embeddings, size, counter = [], [], 0, 0
    for documents, labels in tqdm(data_iter):
        embed = inference_one_batch(documents)
        text_lists.extend(labels)
        embeddings.append(embed)
        size += len(embed)
        if size > args['cut_size']:
            embed = torch.cat(embeddings)
            torch.save((text_lists, embed), f'dpr_chunk_{args["local_rank"]}_{counter}.pt')
            counter += 1
            embeddings = []
    if len(embed) > 0:
        embed = torch.cat(embeddings)
        torch.save((text_lists, embed), f'dpr_chunk_{args["local_rank"]}_{counter}.pt')

if __name__ == "__main__":
    tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    
    args = vars(parser_args())
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    model.cuda()
    model.eval()

    inference(**args)
    torch.distributed.barrier()

