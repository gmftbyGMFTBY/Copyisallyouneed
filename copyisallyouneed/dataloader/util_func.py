import torch
import ipdb
from copy import deepcopy
import random
import requests
import json
from itertools import (takewhile, repeat, islice)


def modify_sentence(ids, min_change=2, prob=0.1, k=2):
    def _random_deletion(rids):
        num_deletion = max(min_change, int(prob*len(rids)))
        delete_idx = random.sample(range(len(rids)), num_deletion)
        n_ids = [rids[i] for i in range(len(rids)) if i not in delete_idx]
        return n_ids
    def _random_swap(rids):
        num_swap = max(min_change, int(prob*len(rids)))
        swap_idx = [random.sample(range(len(rids)), 2) for _ in range(num_swap)]
        n_ids = deepcopy(rids)
        for i, j in swap_idx:
            n_ids[i], n_ids[j] = n_ids[j], n_ids[i]
        return n_ids
    def _random_duplicate(rids):
        # 1-gram or 2-gram
        num_duplicate = max(min_change, int(prob*len(rids)))
        duplicate_idx = random.sample(range(len(rids)-1), num_duplicate)
        n_rids = []
        for idx, i in enumerate(rids):
            if idx in duplicate_idx:
                if random.random() > 0.5:
                    # 2-gram
                    n_rids.extend([rids[idx], rids[idx+1], rids[idx], rids[idx+1]])
                else:
                    n_rids.extend([rids[idx], rids[idx]])
            else:
                n_rids.append(i)
        return n_rids
    rest = []
    for _ in range(k):
        rids = _random_deletion(ids)
        rids = _random_swap(rids)
        rids = _random_duplicate(rids)
        rest.append(rids)
    return rest


def truncate_pair_with_other_ids(cids, rids, tcids, trids, scids, srids, max_length):
    # change the cids and rids in place
    max_length -= 3    # [CLS], [SEP], [SEP]
    while True:
        l = len(cids) + len(rids)
        if l <= max_length:
            break
        if len(cids) > 2 * len(rids):
            cids.pop(0)
            tcids.pop(0)
            scids.pop(0)
        else:
            rids.pop()
            trids.pop()
            srids.pop()


def truncate_pair_with_labels(cids, cids_labels, rids, max_length, rids_labels=None):
    # change the cids and rids in place
    max_length -= 3    # [CLS], [SEP], [SEP]
    while True:
        l = len(cids) + len(rids)
        if l <= max_length:
            break
        if len(cids) > 2 * len(rids):
            cids.pop(0)
            cids_labels.pop(0)
        else:
            rids.pop()
            if rids_labels:
                rids_labels.pop()


def truncate_pair(cids, rids, max_length):
    # change the cids and rids in place
    max_length -= 3    # [CLS], [SEP], [SEP]
    while True:
        l = len(cids) + len(rids)
        if l <= max_length:
            break
        if len(cids) > 2 * len(rids):
            cids.pop(0)
        else:
            rids.pop()


def truncate_pair_two_candidates(cids, rids1, rids2, max_length, sids=None):
    max_length -= 4    # [CLS] ctx [SEP] rids1 [SEP] rids2 [SEP]
    while True:
        l = len(cids) + len(rids1) + len(rids2)
        if l <= max_length:
            break
        if len(cids) > len(rids1) + len(rids2):
            cids.pop(0)
            if sids:
                sids.pop(0)
        elif len(rids1) > len(rids2):
            rids1.pop()
        else:
            rids2.pop()


def generate_mask(ids, pad_token_idx=0):
    '''generate the mask matrix of the ids, default padding token idx is 0'''
    mask = torch.ones_like(ids)
    mask[ids == pad_token_idx] = 0.
    return mask
    # attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
    # attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
    # attn_mask = torch.zeros_like(ids)
    # attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
    # return attn_mask


def to_cuda(*args):
    '''map the tensor on cuda device'''
    if not torch.cuda.is_available():
        return args
    tensor = []
    for i in args:
        i = i.cuda()
        tensor.append(i)
    return tensor


def mask_sentence(
        ids, min_mask_num, max_mask_num, masked_lm_prob, 
        special_tokens=[], mask=-1, vocab_size=21128,
    ):
    '''change the ids, and return the mask_label'''
    num_valid = len([i for i in ids if i not in special_tokens])
    num_mask = max(
        min_mask_num,
        min(
            int(masked_lm_prob * num_valid),
            max_mask_num,
        )
    )

    mask_pos = [idx for idx, i in enumerate(ids) if i not in special_tokens]
    mask_idx = random.sample(mask_pos, num_mask)
    mask_label = []
    for idx, i in enumerate(ids):
        if idx in mask_idx:
            ratio = random.random()
            if ratio < 0.8:
                ids[idx] = mask
            elif ratio < 0.9:
                # random change
                ids[idx] = random.choice(list(range(vocab_size)))
            mask_label.append(i)
        else:
            mask_label.append(-1)
    return mask_label

# ========== dual-bert ========== #
def length_limit(ids, max_len):
    '''the first token must be [CLS]'''
    if len(ids) > max_len:
        ids = [ids[0]] + ids[-(max_len-1):]
    return ids

def length_limit_res(ids, max_len, sep=0):
    '''the last token must be [SEP], and the first token must be [CLS]'''
    if len(ids) > max_len:
        ids = ids[:max_len-1] + [sep]
    return ids

# ======== Evaluation Perturbation ========== # 
def delete(ids, tids, delete_ratio=0.15, min_delete_num=2, special_tokens=[]):
    delete_num = max(
        min_delete_num,
        min(
            len(ids),
            int(len(ids) * delete_ratio),
        )
    )
    delete_idx = [i for i in range(len(ids)) if ids[i] not in special_tokens]
    delete_idx = random.sample(delete_idx, delete_num)

    new_ids, delete_label, new_tids = [], [], []
    for i in ids:
        if i not in delete_idx:
            new_ids.append(i)
            delete_label.append(-1)
        else:
            delete_label.append(len(new_ids))
    pert_label = [-1 if i == -1 else 0 for i in delete_label]
    return new_ids, delete_label, pert_label

def duplicate(ids, duplicate_ratio=0.15, min_duplicate_num=2, special_tokens=[]):
    duplicate_num = max(
        min_duplicate_num,
        min(
            len(ids),
            int(len(ids) * duplicate_ratio),
        )
    )
    duplicate_idx = [i for i in range(len(ids)) if ids[i] not in special_tokens]
    duplicate_idx = random.sample(duplicate_idx, duplicate_num)

    new_ids, duplicate_label = [], []
    for i in ids:
        if i not in duplicate_idx:
            new_ids.append(i)
            duplicate_label.append(-1)
        else:
            num = random.choice([2, 3, 4])
            new_ids.extend([i] * num)
            duplicate_label.extend([len(new_ids)-i_ for i_ in range(num)])
    pert_label = [-1 if i == -1 else 1 for i in duplicate_label]
    return new_ids, duplicate_label, pert_label


def replacement(ids, replace_ratio=0.15, min_replace_num=2, vocab_size=0, special_tokens=[]):
    replace_num = max(
        min_replace_num,
        min(
            len(ids),
            int(len(ids) * replace_ratio),
        )
    )
    replace_idx = [i for i in range(len(ids)) if ids[i] not in special_tokens]
    replace_idx = random.sample(replace_idx, replace_num)

    new_ids, replace_label = [], []
    for i in ids:
        if i not in replace_idx:
            new_ids.append(i)
            replace_label.append(-1)
        else:
            # random replace
            new_ids.append(random.choice(range(vocab_size)))
            replace_label.append(i)
    pert_label = [-1 if i == -1 else 2 for i in replace_label]
    return new_ids, replace_label, pert_label


def mask_sentence_only_mask(
        ids, min_mask_num, max_mask_num, masked_lm_prob, 
        special_tokens=[], mask=-1, vocab_size=21128,
    ):
    '''change the ids, and return the mask_label'''
    num_valid = len([i for i in ids if i not in special_tokens])
    num_mask = max(
        min_mask_num,
        min(
            int(masked_lm_prob * num_valid),
            max_mask_num,
        )
    )
    mask_pos = [idx for idx, i in enumerate(ids) if i not in special_tokens]
    mask_idx = random.sample(mask_pos, num_mask)
    mask_label = []
    for idx, i in enumerate(ids):
        if idx in mask_idx:
            ids[idx] = mask
            mask_label.append(i)
        else:
            mask_label.append(-1)
    return mask_label

# ========== context augmentation ========== #
def sentence_shuffle(context_utterances):
    if len(context_utterances) == 1:
        return context_utterances
    else:
        random_idx = list(range(len(context_utterances)))
        while True:
            random.shuffle(random_idx)
            if random_idx[-1] != len(context_utterances) - 1:
                break
        context_utterances = [context_utterances[i] for i in random_idx]
        return context_utterances

def token_shuffle(context_utterances):
    for i in range(len(context_utterances)):
        random.shuffle(context_utterances[i])
    return context_utterances

def sentence_deletion(context_utterances):
    if len(context_utterances) == 1:
        return context_utterances
    else:
        random_idx = random.choice(range(len(context_utterances)-1))
        context_utterances = [context_utterances[i] for i in range(len(context_utterances)) if i != random_idx]
        return context_utterances

def replace_last_utterance(context_utterances, pool):
    response = random.choice(pool)['rids']
    response = response[1:-1]
    context_utterances[-1] = response
    return context_utterances

def random_insert_before_context(context_utterances, pool):
    u = random.choice(random.choice(pool)['cids'])
    context_utterances.insert(0, u)
    return context_utterances

def random_insert_context(context_utterances, pool):
    u = random.choice(random.choice(pool)['cids'])
    idx = random.choice(range(len(context_utterances)))
    context_utterances.insert(idx, u)
    return context_utterances


# texsmart chinese tokenization
def texsmart_segmentation(engine, text, useful_pos_tag=None):
    output = engine.parse_text(text)
    seg_sentence = []
    for each_word in output.phrases():
        # if each_word.tag in useful_pos_tag:
        seg_sentence.append(each_word.str)
    return seg_sentence

# count lines of the large file
def iter_count(file_name):
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

# iter load the lines
def load_lines_chunk(file, num_lines):
    next_n_lines = list(islice(file, num_lines))
    return next_n_lines
