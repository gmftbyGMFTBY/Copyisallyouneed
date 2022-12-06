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
        # TODO: fully fine-tuned
        # for name, param in self.phrase_encoder.named_parameters():
        #     if 'encoder.layer.11' not in name:
        #         param.requires_grad = False
        # print(f'[!] only the last BERT layer is fine-tuned')
        
        # model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.args['prefix_encoder_tokenizer'][self.args['lang']])
        self.vocab_size = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.args['lang'] == 'zh' else self.tokenizer.bos_token_id

        self.model = GPT2LMHeadModel.from_pretrained(self.args['prefix_encoder_model'][self.args['lang']])
        self.token_embeddings = nn.Parameter(list(self.model.lm_head.parameters())[0])
        # MLP: mapping bert phrase start representations
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
        )
        # MLP: mapping bert phrase end representations
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size//2)
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
        # TODO: inner loss function remove the temperature factor
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
        ids, ids_mask = batch['gpt2_ids'], batch['gpt2_mask']
        outputs = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)[-1]
        # get token loss
        loss_0, acc_0 = self.get_token_loss(ids, last_hidden_states, ids_mask)

        ## encode the document with the BERT encoder model
        dids, dids_mask = batch['bert_ids'], batch['bert_mask']
        dindex_s, dindex_e = batch['phrase_start_index'], batch['phrase_end_index']
        phrase_to_doc = batch['phrase_to_doc']
        phrase_num = len(phrase_to_doc)
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        # get the document representations
        s_rep = self.s_proj(output)    # [B_doc, S_doc, 768//2] => [B_doc*S_doc, 768//2]
        e_rep = self.e_proj(output)    

        # collect the phrase start representations and phrase end representations
        s_rep = s_rep.reshape(-1, s_rep.size(-1))
        e_rep = e_rep.reshape(-1, e_rep.size(-1))    # [B_doc*S_doc, 768//2]

        # collect the query representations
        query = last_hidden_states.reshape(-1, last_hidden_states.size(-1))
        query_start = query[:, :self.model.config.hidden_size//2]
        query_end = query[:, self.model.config.hidden_size//2:]

        # training the representations of the end tokens
        candidate_reps = torch.cat([
            self.token_embeddings[:, :self.model.config.hidden_size], 
            s_rep], dim=0
        )
        logits = torch.matmul(query_start, candidate_reps.t())    # [Q, B*]   
        logits /= self.args['temp']
        logits[phrase_indexes] += mask_pos_matrix
        logits[phrase_indexes] += mask_pad_matrix
        # mask the start token of the phrase
        logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e4
        mask = torch.zeros_like(logits)
        mask[phrase_pos_index, start_pos] = 1.
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss_1 = (-loss_.sum(dim=-1)).sum() / valid_num
        acc_3 = logits[phrase_pos_index].max(dim=-1)[1] == start_pos
        acc_3 = acc_3.to(torch.float).mean().item()

        # training the representations of the start tokens
        candidate_reps = torch.cat([
            self.token_embeddings[:, self.model.config.hidden_size:], 
            e_rep], dim=0
        )
        logits = torch.matmul(query_end, candidate_reps.t())    # [Q, B*]  
        logits /= self.args['temp']
        logits[phrase_indexes] += mask_pos_matrix
        logits[phrase_indexes] += mask_pad_matrix
        logits[phrase_pos_index, token_labels[phrase_pos_index]] = -1e4
        mask = torch.zeros_like(logits)
        mask[phrase_pos_index, end_pos] = 1.
        valid_num = phrase_pos_index.sum().item()
        loss_ = F.log_softmax(logits, dim=-1) * mask
        loss_4 = (-loss_.sum(dim=-1)).sum() / valid_num
        acc_4 = logits[phrase_pos_index].max(dim=-1)[1] == end_pos
        acc_4 = acc_4.to(torch.float).mean().item()
        return (
            loss_0,     # token loss
            loss_1,     # token-head loss
            loss_2,     # token-tail loss
            acc_0,      # token accuracy
            acc_1,      # token-head accuracy
            acc_2       # phrase-tail accuracy
        )

    def make_position_mask(self, mask_pos, length):
        mask_pos_matrix = torch.zeros(len(mask_pos), self.vocab_size + length).cuda()
        mask_pos_matrix[:, self.vocab_size:] = -1e4
        counting = self.vocab_size
        for i, m in enumerate(mask_pos):
            mask_pos_matrix[i, counting:counting+m] = 0
            counting += m 
        return mask_pos_matrix

    def make_padding_mask(self, attention_mask):
        mask_pad_matrix = attention_mask.reshape(1, -1).to(torch.float)   # [B_doc*S_doc]
        mask_pad_matrix = torch.where(mask_pad_matrix == 1, 0.0, -1e4)
        mask_pad_matrix = torch.cat((torch.zeros(1, self.vocab_size).cuda(), mask_pad_matrix), dim=-1)
        return mask_pad_matrix

