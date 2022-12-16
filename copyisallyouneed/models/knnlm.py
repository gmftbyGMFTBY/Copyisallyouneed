from header import *
import sys
sys.path.append('../data')
from dpr_1024 import *
from .agent import top_k_top_p_filtering


class KNNLMBaseline(nn.Module):

    def __init__(self, **args):
        super(KNNLMBaseline, self).__init__()
        self.args = args
        self.vocab = AutoTokenizer.from_pretrained(args['tokenizer'][args['lang']])
        self.model = GPT2LMHeadModel.from_pretrained(args['pretrained_model'][args['lang']])
        self.pad = self.vocab.eos_token_id
        self.unk = self.vocab.unk_token_id
        self.special_tokens = set([self.pad])
        self.gen_loss_fct = nn.NLLLoss(ignore_index=self.vocab.eos_token_id, reduction='none')

        if self.args['mode'] == 'test':
            self.searcher = Searcher('IVF10000,PQ16', dimension=768, nprobe=1)
            self.searcher.load(f'{args["root_dir"]}/data/wikitext103_1024/knnlm/knnlm_faiss.ckpt', f'{args["root_dir"]}/data/wikitext103_1024/knnlm/knnlm_corpus.ckpt')
            # move to the gpu and speedup the searching
            self.searcher.move_to_gpu(0)

    @torch.no_grad()
    def calculate_ppl(self, batch):
        self.model.eval()
        ids, ids_mask = batch['ids'], batch['ids_mask']
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
            cands, dists = self.searcher._search(
                sub_hidden.cpu().numpy(), 
                topk=self.args['search_topk']
            )
            cands = torch.LongTensor([[int(i) for i in j] for j in cands]).unsqueeze(-1).cuda()
            dists = torch.tensor(dists).cuda()    # [S, K]
            dists = F.softmax(-dists, dim=-1).unsqueeze(-1)   # [S, K, 1]
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
    def generate_new_logits(self, logits, hidden, topk=10, temp=100, ids=None):
        cands, dists = self.searcher._search(hidden.unsqueeze(0).cpu().numpy(), topk=topk)
        cands = torch.LongTensor([int(i) for i in cands[0]]).cuda()
        dists = torch.tensor(dists[0]).cuda()
        knn_logits = torch.zeros(topk, len(self.vocab)).cuda()    # [K, V]
        knn_logits[range(topk), cands] = F.softmax(-dists, dim=-1)
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
    def nucleus_sampling(self, ids, max_length, top_p):
        generated = []
        past_key_values = None
        for _ in range(max_length):
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = output['past_key_values']
            hidden = output['hidden_states'][-1][-1, -1, :]
            next_token_logits = output['logits'][-1, -1, :]
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['search_topk'], ids=ids)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=0, top_p=top_p, filter_value=0)
            filtered_logits[self.unk] = 0
            # no softmax for nucleus sampling
            ids = torch.multinomial(filtered_logits, num_samples=1).reshape(1, 1)
            generated.append(ids.item())
        string = self.vocab.decode(generated)
        return string

    @torch.no_grad()
    def forward(self, batch):
        self.model.eval()
        ids, ids_mask, vl = batch['ids'], batch['ids_mask'], batch['vl']
        output = self.model(input_ids=ids, attention_mask=ids_mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
        collection_rep, collection_target = [], []
        ids = ids.tolist()
        for rep, ids_, l in zip(output, ids, vl):
            collection_rep.append(rep[:l-1, :])
            collection_target.extend(ids_[1:l])
        collection_rep = torch.cat(collection_rep).cpu()
        assert len(collection_rep) == len(collection_target)
        collection_target = [str(i) for i in collection_target]
        return collection_rep, collection_target

    @torch.no_grad()
    def greedy_search(self, ids, max_length):
        generated = []
        past_key_values = None
        for _ in range(max_length):
            output = self.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = output['past_key_values']
            hidden = output['hidden_states'][-1][-1, -1, :]
            next_token_logits = output['logits'][-1, -1, :]
            next_token_logits = self.generate_new_logits(next_token_logits, hidden, topk=self.args['search_topk'], ids=ids)
            next_token_logits[self.unk] = -np.inf
            ids = torch.argmax(next_token_logits, dim=-1).reshape(1, 1)
            generated.append(ids.item())
        string = self.vocab.decode(generated)
        return string


