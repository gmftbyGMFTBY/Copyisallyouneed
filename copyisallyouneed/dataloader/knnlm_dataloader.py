from header import *
from .util_func import *


class KNNLMInferenceDataset(Dataset):

    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'][args['lang']])
        path = f'{args["data_root_dir"]}/base_data_128.txt'
        self.vocab.pad_token = self.vocab.eos_token
        self.pad = self.vocab.eos_token_id

        self.data = []
        counter = 0
        with open(path) as f:
            pbar = tqdm(f.readlines())
            for line in pbar: 
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                tokens = self.vocab.encode(chunk, add_special_tokens=False)[:self.args['max_len']]
                if len(tokens) > 32:
                    self.data.append(chunk)
                    counter += len(tokens)
                    # if counter >= 103000000:
                    #     break
                pbar.set_description(f'[!] collect key-values: {counter}')
        print(f'[!] collect {len(self.data)} samples and {counter} key-values')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate(self, batch):
        item = self.vocab(batch, padding=True)
        ids = torch.LongTensor(item['input_ids']).cuda()
        mask = torch.LongTensor(item['attention_mask']).cuda()
        vl = mask.sum(dim=-1)
        return {
            'ids': ids, 
            'ids_mask': mask, 
            'vl': vl
        }
