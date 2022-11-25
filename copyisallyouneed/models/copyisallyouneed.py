from header import *

class Copyisallyouneed(nn.Module):

    def __init__(self, **args):
        super(Copyisallyouneed, self).__init__()
        self.args = args

        # bert-encoder model
        self.phrase_encoder = AutoModel.from_pretrained(
            self.args['phrase_encoder_model'][self.args['lang']]
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            self.args['phrase_encoder_tokenizer'][self.args['lang']]
        )
        
        # only fine-tune the last transformer layer parameters
        # for name, param in self.phrase_encoder.named_parameters():
        #     if 'encoder.layer.11' not in name:
        #         param.requires_grad = False
        # print(f'[!] only the last BERT layer is fine-tuned')
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']])
        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.pad = self.tokenizer.bos_token_id

        self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']])
        self.token_embeddings = nn.Parameter(torch.randn((len(self.tokenizer), 768*2)))
        self.h_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size*2)
        )
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        )
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def get_query_rep(self, ids):
        self.eval()
        output = self.model(input_ids=ids, output_hidden_states=True)['hidden_states'][-1][:, -1, :]
        return self.h_proj(output)

    def get_token_loss(self, ids, hs, ids_mask):
        # no pad token
        label = ids[:, 1:]
        logits = torch.matmul(
            hs[:, :-1, :],
            self.token_embeddings.t()
        )
        logits /= self.args['temp']
        loss = self.gen_loss_fct(logits.view(-1, logits.size(-1)), label.reshape(-1))
        chosen_tokens = torch.max(logits, dim=-1)[1]
        gen_acc = (chosen_tokens.reshape(-1) == label.reshape(-1)).to(torch.long)
        valid_mask = (label != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc

    def forward(self, batch):
        ## gpt2 query encoder
        ids, ids_mask = batch['ids'], batch['ids_mask']
        bsz, seqlen = ids.size()
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        last_hidden_states = self.h_proj(outputs.hidden_states[-1])
        # loss_0, acc_0 = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## bert
        dids, dids_mask = batch['dids'], batch['dids_mask']
        dindex_s, dindex_e = batch['dindex_s'], batch['dindex_e']
        with torch.no_grad():
            output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
            doc_bsz, seqlen, _ = output.size()
            doc_vl = dids_mask.sum(dim=-1)

        # extract the contextual embeddings
        s_rep = self.s_proj(output)    # [B, S, E]
        e_rep = self.e_proj(output)    # [B, S, E]
        start_embeddings, end_embeddings = [], []
        token_start_embeddings, token_end_embeddings = [], []
        start_pos, end_pos = [], []
        mask_pos = []
        counter = self.vocab_size
        for idx in range(doc_bsz):
            vl = doc_vl[idx]
            s = dindex_s[idx]
            e = dindex_e[idx]
            start_embeddings.append(s_rep[idx][:vl])
            start_pos.append(counter+s)
            end_embeddings.append(e_rep[idx][:vl])
            end_pos.append(counter+e)
            counter += vl
            mask_pos.append(vl.item())
        start_embeddings = torch.cat(start_embeddings)
        end_embeddings = torch.cat(end_embeddings)
        start_pos = torch.LongTensor(start_pos).cuda()
        end_pos = torch.LongTensor(end_pos).cuda()

        token_start_embeddings = s_rep[range(doc_bsz), dindex_s]
        token_end_embeddings = s_rep[range(doc_bsz), dindex_e]
        token_start_pos = torch.LongTensor(range(doc_bsz)).cuda() + self.vocab_size
        token_end_pos = torch.LongTensor(range(doc_bsz)).cuda() + self.vocab_size

        # extract query represetnation
        pos_index = batch['pos_ids']    # [B_p]
        vl = batch['vl']

        # make the mask_pos
        mask_pos_matrix = torch.zeros(len(mask_pos), self.vocab_size + len(start_embeddings)).cuda()
        mask_pos_matrix[:, self.vocab_size:] = -1e4
        counting = self.vocab_size
        for i, m in enumerate(mask_pos):
            mask_pos_matrix[i, counting:counting+m] = 0
            counting += m 

        #### training the query head
        query_reps, token_labels, counters, counter, cc = [], [], [], 0, 0
        for ids_, hn, pos_list, l in zip(ids, last_hidden_states, pos_index, vl):
            query_reps.append(hn[:l-1])
            token_labels.append(ids_[1:l])
            for i in pos_list:
                counters.append((cc+i, self.vocab_size + counter))
                counter += 1
            cc += l - 1

        query_reps = torch.cat(query_reps)
        start_query_reps = query_reps[:, :self.model.config.hidden_size]
        end_query_reps = query_reps[:, self.model.config.hidden_size:]
        token_labels = torch.cat(token_labels)
        phrase_labels = torch.ones(cc).fill_(-1).to(torch.long)
        phrase_labels[[i for i, _ in counters]] = torch.LongTensor([j for _, j in counters])
        phrase_labels = phrase_labels.cuda()
        
        phrase_pos_index = phrase_labels != -1
        token_pos_index = phrase_labels == -1

        # training the token-level head
        candidate_reps = torch.cat([
            self.token_embeddings[:, :self.model.config.hidden_size], 
            token_start_embeddings,
            ], dim=0
        )
        logits = torch.matmul(start_query_reps, candidate_reps.t())
        logits /= self.args['temp']
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), token_labels] = 1.
        logits[phrase_pos_index, token_start_pos] = -1e3
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss_1 = (-loss_.sum(dim=1)).mean()
        acc_1 = logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        acc_1 = acc_1.to(torch.float).mean().item()

        # training the token-level end
        candidate_reps = torch.cat([
            self.token_embeddings[:, self.model.config.hidden_size:], 
            token_end_embeddings,
            ], dim=0
        )
        logits = torch.matmul(end_query_reps, candidate_reps.t())
        logits /= self.args['temp']
        mask = torch.zeros_like(logits)
        mask[range(len(logits)), token_labels] = 1.
        logits[phrase_pos_index, token_end_pos] = -1e3
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss_2 = (-loss_.sum(dim=1)).mean()
        acc_2 = logits[token_pos_index].max(dim=-1)[1] == token_labels[token_pos_index]
        acc_2 = acc_2.to(torch.float).mean().item()

        # training the phrase-leve head
        candidate_reps = torch.cat([
            self.token_embeddings[:, :self.model.config.hidden_size], 
            start_embeddings], dim=0
        )
        logits = torch.matmul(start_query_reps, candidate_reps.t())    # [Q, B*]   
        logits /= self.args['temp']
        mask = torch.zeros_like(logits)
        mask[phrase_pos_index, start_pos] = 1.
        
        counting = self.vocab_size
        phrase_indexes = phrase_pos_index.to(torch.long).nonzero().squeeze(dim=-1)
        assert len(phrase_indexes) == len(mask_pos)
        logits[phrase_indexes] += mask_pos_matrix
        logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e4
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss_3 = (-loss_.sum(dim=-1)).sum() / valid_num
        acc_3 = logits[phrase_pos_index].max(dim=-1)[1] == start_pos
        acc_3 = acc_3.to(torch.float).mean().item()

        # training the phrase-leve tail
        candidate_reps = torch.cat([
            self.token_embeddings[:, self.model.config.hidden_size:], 
            end_embeddings], dim=0
        )
        logits = torch.matmul(end_query_reps, candidate_reps.t())    # [Q, B*]  
        logits /= self.args['temp']
        
        counting = self.vocab_size
        phrase_indexes = phrase_pos_index.to(torch.long).nonzero().squeeze(dim=-1)
        assert len(phrase_indexes) == len(mask_pos)
        logits[phrase_indexes] += mask_pos_matrix
        logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e4
        
        mask = torch.zeros_like(logits)
        mask[phrase_pos_index, end_pos] = 1.
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss_4 = (-loss_.sum(dim=-1)).sum() / valid_num
        acc_4 = logits[phrase_pos_index].max(dim=-1)[1] == end_pos
        acc_4 = acc_4.to(torch.float).mean().item()
        return loss_1, loss_2, loss_3, loss_4, acc_1, acc_2, acc_3, acc_4

    @torch.no_grad()
    def fast_rerank(self, ids, candidates, temp=1.0):
        self.model.eval()
        # 1. tokenize candidates
        tokens = self.tokenizer.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
        # 2. prepare the ids and mask
        cids = [torch.LongTensor(t) for t in tokens]
        cids = pad_sequence(cids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(cids, pad_token_idx=self.pad)
        cids, mask = to_cuda(cids, mask)
        ids = ids.expand(len(cids), -1)
        seqlen = ids.size(-1)
        mask = torch.cat([torch.ones_like(ids), mask], dim=-1)
        ids = torch.cat([ids, cids], dim=-1)
        # 3. gpt2 encoding
        hidden_state = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        # 4. calculating the confidence 
        shift_label = ids[:, seqlen:]
        shift_hidden = hidden_state[:, seqlen-1:-1, :]
        shift_logits = torch.matmul(shift_hidden, self.token_embeddings.t())    # [B, S, V]
        confidence = torch.gather(shift_logits, 2, shift_label.unsqueeze(-1)).squeeze(-1)    # [B, S]
        vl = mask.sum(dim=-1) - seqlen
        confidence = torch.stack([c[:l].mean() for c, l in zip(confidence, vl)])    # [B]
        # confidence = torch.stack([c[:l].min() for c, l in zip(confidence, vl)])    # [B]
        return F.softmax(confidence/temp, dim=-1)

