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

    @torch.no_grad()
    def calculate_ppl(self, batch):
        self.model.eval()
        ids, ids_mask = batch['ids'], batch['ids_mask']
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask).logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

