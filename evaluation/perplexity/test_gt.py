from tqdm import tqdm
from evaluate import load
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.cuda.amp import autocast
import ipdb
import mauve
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
import argparse
from datasets import load_dataset

'''test the ground-truth perplexity:
refer to https://huggingface.co/docs/transformers/perplexity for more details
'''

def calculate_ppl(test):
    device = "cuda"
    model_id = "gpt2-large"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    encodings = tokenizer("\n\n".join(test), return_tensors="pt")
    max_length = model.config.n_positions
    stride = 1024    # maximum position
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl

if __name__ == "__main__":
    test = load_dataset("wikitext", "wikitext-103-v1", split="test")
    test = [line['text'] for line in test]
    ppl = calculate_ppl(test)
    print(f'[!] ppl: {ppl}')


