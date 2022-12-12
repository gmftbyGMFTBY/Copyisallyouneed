from header import *
from .util_func import *


class KNNLMInferenceDataset(Dataset):

    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'][args['lang']])
        path = f'{args["data_root_dir"]}/base_data_128.txt'
        self.pad = self.vocab.eos_token_id

        self.data = []
        counter = 0
        with open(path) as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                tokens = self.vocab.encode(chunk, add_special_tokens=False)[:self.args['max_len']]
                self.data.append(tokens)
                counter += len(tokens)
                if counter >= 10000000:
                    break
        print(f'[!] collect {len(self.data)} samples and {counter} key-values')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = self.data[i]
        return torch.LongTensor(ids)

    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids, pad_token_idx=self.pad)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
        }
