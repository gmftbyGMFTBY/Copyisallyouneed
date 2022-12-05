from header import *
from .util_func import *


class CopyisallyouneedWikitext103Dataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_encoder_tokenizer'][args['lang']])
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])
        self.data_root_path = args['data_root_dir']
        self.file_lists = [f'{self.data_root_path}/dpr_search_result_128_{i}.txt' for i in range(1)]
        # count the number of the samples
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['global_rank']
            random.seed(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker {self.args["local_rank"]}:')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        # load the base_data
        base_data = {}
        with open(f'{self.data_root_path}/base_data_128.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1].strip()
                if id_label:
                    base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over')

    def __len__(self):
        return self.size

    def load_one_chunk(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())
            base_index = item['index']

            # collect one document
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                length_s = len(item_)
                # replace the <unk> with <|endoftext|>
                item_ = item_.replace('< |endoftext| >', '<|endoftext|>')
                if counter > 0:
                    item_ = ' ' + item_

                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid_, doc_pos = docid[0]
                    if counter > 0:
                        docs.append((counter - 1, length_s, len(items), docid_, doc_pos))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            while len(docs) > 0 and counter - docs[-1][0] <= 3:
                docs.pop()

            if len(ids) > 0:
                ids_total.append(torch.LongTensor(ids))
                vl.append(len(ids))
            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index = []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                
                # bert-base-cased UNK replacement
                phrase = phrase.replace('< |endoftext| >', '[UNK]')
                pre_phrase = pre_phrase.replace('< |endoftext| >', '[UNK]')
                post_phrase = post_phrase.replace('< |endoftext| >', '[UNK]')

                phrase_ids, pre_phrase_ids, post_phrase_ids = self.bert_vocab.batch_encode_plus([
                    phrase, pre_phrase, post_phrase
                    ], add_special_tokens=False)['input_ids']
                self._truncate_triplet(
                    pre_phrase_ids, 
                    phrase_ids, 
                    post_phrase_ids, 
                    self.args['doc_max_length'] - 2
                )
                # special case for the phrase in the prefix
                if base_index == docid:
                    doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.cls_token_id]
                else:
                    doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.sep_token_id]
                doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), len(pre_phrase_ids) + len(phrase_ids)
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append((doc_s_pos, doc_e_pos))
                pos_index.append(pos_in_ids)
            pos_index_total.append(pos_index)
        return ids_total, doc_ids, doc_index, pos_index_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
        dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
        ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
        }


class CopyisallyouneedChineseDataset(Dataset):
    
    def __init__(self, **args):
        self.args = args
        self.bert_vocab = AutoTokenizer.from_pretrained(args['phrase_encoder_tokenizer'][args['lang']])
        self.vocab = AutoTokenizer.from_pretrained(args['prefix_encoder_tokenizer'][args['lang']])

        self.data_root_path = args['data_root_dir']
        file_num = 1
        self.file_lists = [f'{self.data_root_path}/bm25_search_result_{i}.txt' for i in range(file_num)]
        self.size = 0
        for path in self.file_lists:
            self.size += iter_count(path)

        if self.args['mode'] == 'train':
            new_seed = args['seed'] + args['global_rank']
            random.seed(new_seed)
            random.shuffle(self.file_lists)
            print(f'[!] file list for worker {self.args["local_rank"]}:')
            print(self.file_lists)

        self.current_file_index = 0
        self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
        self.cache = []
        self.buffer_size = args['buffer_size']
        self.if_last_over = True
        self.last_delta = 0

        base_data = {}
        with open(f'{self.data_root_path}/base_data.txt') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                chunk = ' '.join(line[:-1])
                id_label = line[-1]
                base_data[id_label] = chunk
        self.base_data = base_data
        print(f'[!] load base data over')

    def __len__(self):
        return self.size

    def load_one_chunk(self):
        assert len(self.cache) == 0
        self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        if len(self.cache) == 0:
            # current file runs over, cyclely loading
            self.current_file_index = 0 if self.current_file_index == len(self.file_lists) - 1 else self.current_file_index + 1
            self.current_file_handler = open(self.file_lists[self.current_file_index], 'r')
            self.cache = load_lines_chunk(self.current_file_handler, self.buffer_size)
        random.shuffle(self.cache)

    def _truncate_triplet(self, a, b, c, max_length):
        while True:
            if len(a) + len(b) + len(c) <= max_length:
                break
            else:
                if len(a) > len(c):
                    a.pop(0)
                else:
                    c.pop()

    def __getitem__(self, i):
        # read till the max bert dids are achieved
        ids_total, vl, doc_ids, doc_index, pos_index_total, pos_index_end_total = [], [], [], [], [], []
        while len(doc_ids) < self.args["max_doc_size"]:
            if len(self.cache) == 0:
                self.load_one_chunk()
            item = json.loads(self.cache[0].strip())
            base_index = item['index']

            # collect documents
            docs, ids, counter, delta_ = [], [], 0, 0
            for item_, docid in item['results'][self.last_delta:]:
                length_s = len(item_)
                item_o = item_
                items = self.vocab.encode(item_, add_special_tokens=False)
                if len(items) == 0:
                    continue
                if len(ids) + len(items) > self.args['max_len']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                if docid:
                    docid = docid[0]
                    if counter > 0 and item_o == self.base_data[docid[0]][docid[1]:docid[1]+length_s]:
                        docs.append((counter - 1, length_s, len(items), docid[0], docid[1], item_o))
                ids.extend(items)
                counter += len(items)
                if len(docs) + len(doc_ids) > self.args['max_doc_size']:
                    self.last_delta += delta_
                    self.if_last_over = False
                    break
                delta_ += 1
            else:
                self.if_last_over = True

            if len(docs) > 0 and counter - docs[-1][0] <= 2:
                docs.pop()
            if len(ids) > 0:
                ids_total.append(torch.LongTensor(ids))
                vl.append(len(ids))
            if self.if_last_over is True:
                self.last_delta = 0
                self.cache.pop(0)

            # encode the documents
            pos_index, pos_index_end = [], []
            for pos_in_ids, length_s, length_i, docid, pos_in_doc, item_o in docs:
                doc_ = self.base_data[docid]
                pre_phrase, post_phrase = doc_[:pos_in_doc], doc_[pos_in_doc+length_s:]
                phrase = doc_[pos_in_doc:pos_in_doc+length_s]
                phrase_ids = self.bert_vocab.encode(phrase, add_special_tokens=False)
                pre_phrase_ids = self.bert_vocab.encode(pre_phrase, add_special_tokens=False)
                post_phrase_ids = self.bert_vocab.encode(post_phrase, add_special_tokens=False)
                try:
                    self._truncate_triplet(pre_phrase_ids, phrase_ids, post_phrase_ids, self.args['doc_max_length'] - 2)
                except:
                    continue
                if base_index == docid:
                    doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.cls_token_id]
                else:
                    doc_ids_ = [self.bert_vocab.cls_token_id] + pre_phrase_ids + phrase_ids + post_phrase_ids + [self.bert_vocab.sep_token_id]
                doc_s_pos, doc_e_pos = 1 + len(pre_phrase_ids), len(pre_phrase_ids) + len(phrase_ids)
                doc_ids.append(torch.LongTensor(doc_ids_))
                doc_index.append((doc_s_pos, doc_e_pos))
                pos_index.append(pos_in_ids)
                pos_index_end.append(pos_in_ids + length_i)
            pos_index_total.append(pos_index)
            pos_index_end_total.append(pos_index_end)
        return ids_total, doc_ids, doc_index, pos_index_total, pos_index_end_total, vl

    def save(self):
        pass
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, dids, dindex, pos_ids, pos_ids_end, vl = batch[0]
        dindex_s = torch.LongTensor([i for i, _ in dindex])
        dindex_e = torch.LongTensor([i for _, i in dindex])
        if self.args['lang'] == 'zh':
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.pad_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids), generate_mask(dids)
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.vocab.eos_token_id)
            dids = pad_sequence(dids, batch_first=True, padding_value=self.bert_vocab.pad_token_id)
            ids_mask, dids_mask = generate_mask(ids, pad_token_idx=self.vocab.eos_token_id), generate_mask(dids, pad_token_idx=self.bert_vocab.pad_token_id)
        ids, dids, ids_mask, dids_mask, dindex_s, dindex_e = to_cuda(ids, dids, ids_mask, dids_mask, dindex_s, dindex_e)
        return {
            'ids': ids, 
            'dids': dids, 
            'vl': vl,
            'ids_mask': ids_mask, 
            'dids_mask': dids_mask, 
            'dindex_s': dindex_s,
            'dindex_e': dindex_e,
            'pos_ids': pos_ids,
            'pos_ids_end': pos_ids_end,
        }
