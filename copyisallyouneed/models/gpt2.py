from header import *

class GPT2Baseline(nn.Module):

    def __init__(self, **args):
        super(GPT2Baseline, self).__init__()
        model = args['pretrained_model'][args['lang']]
        self.model_name = model
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.vocab = AutoTokenizer.from_pretrained(model)
        self.vocab_size = len(self.vocab)
        self.args = args

        self.pad = self.vocab.eos_token_id
        self.unk = self.vocab.unk_token_id
        self.special_tokens = set([self.pad])
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad)

    @torch.no_grad()
    def nucleus_sampling(self, batch):
        '''batch_size is 1'''
        self.model.eval()
        ids = batch['ids']
        generated = []
        while True:
            output = self.model(
                input_ids=ids,
            )[0]    # [1, S, V]
            next_token_logits = output[-1, -1, :]    # [V]
            next_token_logits[self.unk] = -np.inf
            if 'gpt2_english' in self.model_name: 
                next_token_logits[198] = -np.inf
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=self.topk, 
                top_p=self.topp
            )
            next_token = torch.multinomial(
                F.softmax(filtered_logits, dim=-1),
                num_samples=1,
            )
            if len(generated) > self.test_max_len:
                break
            generated.append(next_token.item())
            ids = torch.cat((ids, next_token.unsqueeze(0)), dim=1)    # [1, S+1]
        string = self.vocab.decode(generated)
        return string

    def forward(self, batch):
        ids, ids_mask = batch['ids'], batch['ids_mask']
        ids, ods = ids[:, :-1], ids[:, 1:]
        ids_mask = ids_mask[:, :-1]
        output = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = output.logits
        loss = self.gen_loss_fct(
            gen_logits.view(-1, gen_logits.size(-1)), 
            ods.reshape(-1)
        )
        # token acc
        chosen_tokens = torch.max(gen_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.reshape(-1) == ods.reshape(-1)).to(torch.long)
        valid_mask = (ods != self.pad).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return loss, gen_acc
