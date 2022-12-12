from header import *
from .util_func import *


class GPT2Dataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'][args['lang']])
        self.pad = self.vocab.convert_tokens_to_ids('<|endoftext|>')
        
        self.data = []
        with open(f'{args["data_root_dir"]}/base_data_128.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1].strip()
                if id_label:
                    self.data.append(chunk)
        print(f'[!] load {len(self.data)} samples')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i]
        ids = self.vocab.encode(text, add_special_tokens=False)[:self.args['max_len']]
        ids = torch.LongTensor(ids)
        return ids

    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, mask = to_cuda(ids, mask)
        return {
            'ids': ids, 
            'ids_mask': mask, 
        }


class GPT2PPLDataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'][args['lang']])
        self.pad = self.vocab.convert_tokens_to_ids('<|endoftext|>')
        
        self.data = []
        with open(f'{args["data_root_dir"]}/test.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip()
                if len(line.split()) > 32:
                    self.data.append(line)
        print(f'[!] load {len(self.data)} samples')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i]
        ids = self.vocab.encode(text, add_special_tokens=False)[:self.args['max_len']]
        ids = torch.LongTensor(ids)
        return ids

    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, mask = to_cuda(ids, mask)
        return {
            'ids': ids, 
            'ids_mask': mask, 
        }
