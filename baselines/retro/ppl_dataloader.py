from header import *
from .util_func import *


class GPT2PPLDataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('<|endoftext|>')
        
        self.data = []
        # directly load the standard wikitext-103 test set from the huggingface datasets package
        dataset = load_dataset('wikitext', 'wikitext-103-v1')['test']

        self.data = load_wikitext_data_split(self.vocab, dataset, self.args)
        print(f'[!] load {len(self.data)} samples')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ids = self.data[i]
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
