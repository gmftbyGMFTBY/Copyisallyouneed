from header import *


def top_k_top_p_filtering_knnlm(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        # special cases
        if logits.max().item() > top_p:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        # cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


class KNNLMBaseline(nn.Module):

    def __init__(self, **args):
        super(KNNLMBaseline, self).__init__()
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'][args['lang']])
        self.model = GPT2LMHeadModel.from_pretrained(args['pretrained_model'][args['lang']])
        self.pad = self.vocab.eos_token_id
        self.unk = self.vocab.unk_token_id
        self.special_tokens = set([self.pad])
        self.gen_loss_fct = nn.NLLLoss(ignore_index=self.vocab.pad_token_id, reduction='none')

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask):
        self.model.eval()
        ids, ids_mask, label = ids[:, :-1], ids_mask[:, :-1], ids[:, 1:]
        output = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)
        logits = output.logits.squeeze(0)    # [S, V]
        hidden = output['hidden_states'][-1].view(-1, 768) # [S, E]
        seqlen, _ = hidden.size()

        sub_chunk_size = 32
        losses = []
        for i in range(0, seqlen, sub_chunk_size):
            sub_hidden = hidden[i:i+sub_chunk_size, :]
            sub_seqlen = len(sub_hidden)
            sub_label = label[:, i:i+sub_chunk_size]
            sub_logits = logits[i:i+sub_chunk_size, :]
            cands, dists = self.searcher._search_dis(
                sub_hidden.cpu().numpy(), 
                topk=self.args['search_topk']
            )
            cands = torch.LongTensor([[int(i) for i in j] for j in cands]).unsqueeze(-1).cuda()    # [S, K, 1]
            dists = torch.tensor(dists).cuda()    # [S, K]
            dists = F.softmax(-dists/self.args['temp'], dim=-1).unsqueeze(-1)   # [S, K, 1]
            knn_logits = torch.zeros(sub_seqlen, self.args['search_topk'], len(self.vocab)).cuda()   # [S, K, V]
            knn_logits.scatter_(2, cands, dists)
            knn_logits = knn_logits.sum(dim=1)    # [S, V]
            new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * F.softmax(sub_logits, dim=-1)    # [S, V]
            new_logits = new_logits.log()
            loss = self.gen_loss_fct(new_logits.view(-1, new_logits.size(-1)), sub_label.view(-1))
            losses.append(loss)
        loss = torch.cat(losses).mean()
        return math.exp(loss.item())

    @torch.no_grad()
    def generate_new_logits(self, logits, hidden, topk=10, temp=100):
        # ignored tokens
        # ignored_tokens = set(['198', '2954', '27', '1279', '29'])
        cands, dists = self.searcher._search_dis(
            hidden.unsqueeze(0).cpu().numpy(), 
            topk=topk
        )
        # valid_index = [False if i in ignored_tokens else True for i in cands[0]]
        cands = [int(i) for i in cands[0]]
        counter_num = sum([j for _, j in Counter(cands).most_common(self.args['center_topk'])])
        dists = torch.tensor(dists[0]).cuda()
        # topk = sum(valid_index)
        # if topk <= int(self.args['collapse_rate'] * self.args['search_topk']) or \
        #     counter_num >= int(self.args['center_collapse_rate'] * self.args['search_topk']):
        #     # the searched results collapse, donot rely on it
        #     return F.softmax(logits, dim=-1)
        # else:
        cands = torch.LongTensor(cands).cuda()
        # cands = cands[valid_index]
        # dists = dists[valid_index]
        knn_logits = torch.zeros(topk, len(self.vocab)).cuda()    # [K, V]
        knn_logits[range(topk), cands] = F.softmax(-dists/self.args['temp'], dim=-1)
        knn_logits = knn_logits.sum(dim=0)    # [V]
        new_logits = self.args['lambda'] * knn_logits + (1 - self.args['lambda']) * F.softmax(logits, dim=-1)
        return new_logits

    @torch.no_grad()
    def greedy_search(self, batch):
        self.model.eval()
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True
            )
            hidden = output['hidden_states'][-1][-1, -1, :]    # [H]
            next_token_logits = output['logits'][-1, -1, :]    # [V]
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['search_topk'], temp=self.args['temp'])
            ignored_tokens = [198, 2954, 1279, 27, 29, self.unk]
            next_token_logits[ignored_tokens] = -np.inf

            next_token = next_token_logits.max(dim=-1)[1].unsqueeze(0)
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
            # ids = ids[:, -self.test_max_ctx_len:]
        string = self.vocab.decode(generated)
        return string

    @torch.no_grad()
    def nucleus_sampling(self, batch):
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True
            )
            hidden = output['hidden_states'][-1][-1, -1, :]    # [H]
            next_token_logits = output['logits'][-1, -1, :]    # [V]
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['search_topk'])
            filtered_logits = top_k_top_p_filtering_knnlm(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp
            )
            # ignore some tokens: \n, unk, <, >, eos_token
            # ignored_tokens = [198, 2954, 27, 1279, 29, self.unk]
            # filtered_logits[ignored_tokens] = -np.inf
            filtered_logits[self.unk] = -np.inf
            next_token = torch.multinomial(
                F.softmax(filtered_logits*2, dim=-1),
                num_samples=1,
            )
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            # reconstruct the ids and ids_mask
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
            # ids = ids[:, -self.test_max_ctx_len:]
        string = self.vocab.decode(generated)
        return string

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        ids, ids_mask = batch['ids'], batch['ids_mask']
        output = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        vl = ids_mask.sum(dim=-1)
        collection_rep, collection_target = [], []
        ids = ids.tolist()
        for rep, ids_, l in zip(output, ids, vl):
            collection_rep.append(rep[:l-1, :])
            collection_target.extend(ids_[1:l])
        collection_rep = torch.cat(collection_rep).cpu()
        try:
            assert len(collection_rep) == len(collection_target)
        except:
            ipdb.set_trace()
        collection_target = [str(i) for i in collection_target]
        return collection_rep, collection_target


