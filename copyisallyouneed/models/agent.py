from header import *
import spacy

class Agent:
    
    def __init__(self, model, args):
        super(Agent, self).__init__()
        self.args = args
        self.model = model
        self.load_last_step = None

        if torch.cuda.is_available():
            self.model.cuda()

        if args['mode'] in ['train']:
            self.set_optimizer_scheduler_ddp()
        if args['model'] == 'gpt2':
            self.train_model = self.train_model_gpt2
        # self.load_latest_checkpoint()

    def set_optimizer_scheduler_ddp(self):
        if self.args['mode'] in ['train']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.scaler = GradScaler()
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_step'], 
                num_training_steps=self.args['total_step'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )

    def load_model(self, path):
        if self.args['mode'] == 'train':
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model_state_dict = state_dict['model_state_dict']
            self.model.module.load_state_dict(model_state_dict)
            self.load_last_step = state_dict['step']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            print(f'[!] load the latest model from {path}')
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))['model_state_dict']
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')
    
    def train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        with autocast():
            batch['current_step'] = current_step
            loss_0, loss_1, loss_2, acc_0, phrase_start_acc, phrase_end_acc, token_start_acc, token_end_acc = self.model(batch)
            loss = loss_0 + loss_1 + loss_2
            loss = loss / self.args['iter_to_accumulate']
        self.scaler.scale(loss).backward()
        if (current_step + 1) % self.args['iter_to_accumulate'] == 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()

        if recoder:
            recoder.add_scalar(f'train/Loss', loss.item(), current_step)
            recoder.add_scalar(f'train/pure_token_head_loss', loss_0.item(), current_step)
            recoder.add_scalar(f'train/start_loss', loss_1.item(), current_step)
            recoder.add_scalar(f'train/end_loss', loss_2.item(), current_step)
            recoder.add_scalar(f'train/pure_token_acc', acc_0, current_step)
            recoder.add_scalar(f'train/token_start_acc', token_start_acc, current_step)
            recoder.add_scalar(f'train/token_end_acc', token_end_acc, current_step)
            recoder.add_scalar(f'train/phrase_start_acc', phrase_start_acc, current_step)
            recoder.add_scalar(f'train/phrase_end_acc', phrase_end_acc, current_step)
        pbar.set_description(f'[!] loss(s|e): {round(loss_1.item(), 4)}|{round(loss_2.item(), 4)}; acc: {round((token_start_acc+token_end_acc)/2, 4)}|{round((phrase_start_acc+phrase_end_acc)/2, 4)}')
        pbar.update(1)

    def load_latest_checkpoint(self):
        path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}'
        prefix_name = f'best_{self.args["version"]}_'
        checkpoints = []
        for file in os.listdir(path):
            if prefix_name in file:
                version = file[len(prefix_name):].strip('.pt')
                version = int(version)
                checkpoints.append((file, version))
        if len(checkpoints) == 0:
            print(f'[!] do not find the latest model checkpoints')
            return
        checkpoints = sorted(checkpoints, key=lambda x:x[-1])
        latest_checkpoint, version = checkpoints[-1]
        latest_checkpoint = os.path.join(path, latest_checkpoint)
        self.load_model(latest_checkpoint)
        print(f'[!] train start from step: {version}')

    def save_model_long(self, path, current_step):
        model_state_dict = self.model.module.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        torch.save(
            {
                'model_state_dict' : model_state_dict,
                'scheduler_state_dict': scheduler_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'step': current_step
            }, 
            path
        )
        print(f'[!] save model into {path}')

    @torch.no_grad()
    def debug_generate_one_step_fast(self, ids, phrase_reps, phrase_sources, decoding_method='greedy', temp=1., top_k=0, top_p=0.92):
        self.model.eval()
        query = self.model.get_query_rep(ids)
        score = torch.matmul(query, phrase_reps.t()).squeeze(0)   

        if decoding_method == 'greedy':
            index = score.max(dim=-1)[1].item()
            candidate = phrase_sources[index]
        elif decoding_method == 'nucleus_sampling':
            score = top_k_top_p_filtering(score, top_k=top_k, top_p=top_p)
            index = torch.multinomial(F.softmax(score/temp, dim=-1), num_samples=1).item()
            candidate = phrase_sources[index]
            if type(candidate) == list:
                candidate_string = self.model.bert_tokenizer.decode(candidate)
            else:
                candidate_string = self.model.tokenizer.decode(candidate)

            scores, topk_index = F.softmax(score, dim=-1).topk(self.args['phrase_topk'], dim=-1)
            candidates = [self.model.bert_tokenizer.decode(phrase_sources[idx]) if type(phrase_sources[idx]) == list else self.model.tokenizer.decode(phrase_sources[idx]) for idx in topk_index]
            scores = scores.tolist()
            print(f'[!] current prefix:')
            print(self.model.tokenizer.decode(ids[0]))
            ipdb.set_trace()
        else:
            pass

        # get textual candidate
        if type(candidate) == list:
            candidate = ' ' + self.model.bert_tokenizer.decode(candidate).replace('[UNK]', '<|endoftext|>')
            sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
        else:
            sub_ids = [candidate]
        sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        ids = torch.cat((ids, sub_ids), dim=-1)
        return ids, candidate

    @torch.no_grad()
    def debug_generate_one_sample(self, text, retriever, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        documents = retriever.search([text], self.args['doc_topk'])[0]
        # add the prefix
        # documents = [text] + documents
        phrase_reps, phrase_sources = self.get_phrases_fast(documents)
        candidates = []
        encode_time = 0
        bt = time.time()
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.debug_generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)
            # encode the document prefix
            if len(ids[0]) > 32 and encode_time == 0:
                prefix_phrase_reps, prefix_phrase_sources = self.get_prefix_phrases_fast([self.model.tokenizer.decode(ids[0])])
                phrase_reps = torch.cat([phrase_reps, prefix_phrase_reps], dim=0)
                phrase_sources.extend(prefix_phrase_sources)
                encode_time += 1
        inference_time = time.time() - bt
        if get_time_cost:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, inference_time
        else:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, None

    @torch.no_grad()
    def generate_multiple_sample(self, text, retriever, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False, random_seeds=[], reference=None):
        '''generate multiple samples by using the same set of phrases with differnt random seed'''
        self.model.eval()
        assert decoding_method == 'nucleus_sampling'
        sample_num = len(random_seeds)
        documents = retriever.search([text], self.args['doc_topk'])[0]
        phrase_reps, phrase_sources = self.get_phrases_fast(documents)
        collections = {s: None for s in random_seeds}
            
        ids_original = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids_original[0])
        for i in range(sample_num):
            ids = ids_original.clone()

            torch.manual_seed(random_seeds[i])
            torch.cuda.manual_seed_all(random_seeds[i])
            candidates = []
            encode_time = 0

            bt = time.time()
            while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
                ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
                candidates.append(candidate)
                # encode the document prefix
                if len(ids[0]) > 32 and encode_time == 0:
                    prefix_phrase_reps, prefix_phrase_sources = self.get_prefix_phrases_fast([self.model.tokenizer.decode(ids[0])])
                    phrase_reps = torch.cat([phrase_reps, prefix_phrase_reps], dim=0)
                    phrase_sources.extend(prefix_phrase_sources)
                    encode_time += 1
            inference_time = time.time() - bt
            collections[random_seeds[i]] = {
                'prefix': text,
                'reference': reference,
                'text': self.model.tokenizer.decode(ids[0, prefix_length:]),
                'phrases': candidates
            }
            if get_time_cost:
                collections[random_seeds[i]]['time_cost'] = inference_time
        return collections

    @torch.no_grad()
    def generate_one_sample(self, text, retriever, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        documents = retriever.search([text], self.args['doc_topk'])[0]
        # add the prefix
        # documents = [text] + documents
        phrase_reps, phrase_sources = self.get_phrases_fast(documents)
        candidates = []
        encode_time = 0
        bt = time.time()
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)
            # encode the document prefix
            if len(ids[0]) > 32 and encode_time == 0:
                prefix_phrase_reps, prefix_phrase_sources = self.get_prefix_phrases_fast([self.model.tokenizer.decode(ids[0])])
                phrase_reps = torch.cat([phrase_reps, prefix_phrase_reps], dim=0)
                phrase_sources.extend(prefix_phrase_sources)
                encode_time += 1
        inference_time = time.time() - bt
        if get_time_cost:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, inference_time
        else:
            return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates, None

    @torch.no_grad()
    def generate_one_step_fast(self, ids, phrase_reps, phrase_sources, decoding_method='greedy', temp=1., top_k=0, top_p=0.92):
        self.model.eval()
        query = self.model.get_query_rep(ids)
        score = torch.matmul(query, phrase_reps.t()).squeeze(0)   

        if decoding_method == 'greedy':
            index = score.max(dim=-1)[1].item()
            candidate = phrase_sources[index]
        elif decoding_method == 'nucleus_sampling':
            score = top_k_top_p_filtering(score, top_k=top_k, top_p=top_p)
            index = torch.multinomial(F.softmax(score/temp, dim=-1), num_samples=1).item()
            candidate = phrase_sources[index]
        else:
            pass

        # get textual candidate
        if type(candidate) == list:
            candidate = ' ' + self.model.bert_tokenizer.decode(candidate).replace('[UNK]', '<|endoftext|>')
            sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
        else:
            sub_ids = [candidate]
        sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        ids = torch.cat((ids, sub_ids), dim=-1)
        return ids, candidate

    @torch.no_grad()
    def get_phrases_fast(self, documents):
        self.model.eval()

        # feed the 1024 maybe to big, leading to OOM
        inner_batch_size = 256
        begin_hidden_states, end_hidden_states, vl, doc_ids = [], [], [], []
        for idx in range(0, len(documents), inner_batch_size):
            s_index, e_index = idx, idx + inner_batch_size
            batch_doc = documents[s_index:e_index]
            batch = self.model.bert_tokenizer.batch_encode_plus(batch_doc, padding=True, return_tensors='pt', max_length=200, truncation=True)
            input_ids = batch['input_ids'].cuda()
            mask = batch['attention_mask'].cuda()
            hs = self.model.phrase_encoder(input_ids, mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]
            bhs = self.model.s_proj(hs)
            ehs = self.model.e_proj(hs)
            begin_hidden_states.extend(bhs)
            end_hidden_states.extend(ehs)
            vl.extend(mask.sum(dim=-1))
            doc_ids.extend(input_ids.tolist())
        assert len(end_hidden_states) == len(begin_hidden_states) == len(documents) == len(vl) == len(doc_ids)

        begin_rep, end_rep = [], []
        phrase_sources = []
        phrase_sources_set = set()
        # remove duplication in the phrase tables
        for begin_doc_rep, end_doc_rep, l, doc_id in zip(begin_hidden_states, end_hidden_states, vl, doc_ids):
            s_pos, e_pos = [], []
            # ignore the [CLS] token
            for i in range(1, l-self.args['left_window_size']-1):
                # ignore the [SEP] token
                for j in range(
                    min(i+self.args['left_window_size'], l-2), 
                    min(i+self.args['right_window_size'], l-2)
                ):
                    phrase = doc_id[i:j+1]
                    if tuple(phrase) not in phrase_sources_set:
                        s_pos.append(i)
                        e_pos.append(j)
                        phrase_sources.append(phrase)
                        phrase_sources_set.add(tuple(phrase))
            s_rep = begin_doc_rep[s_pos, :]
            e_rep = end_doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([begin_rep, end_rep], dim=-1)
        phrase_reps = torch.cat([
            phrase_reps,
            self.model.token_embeddings
        ], dim=0)
        phrase_sources.extend([idx for idx in range(len(self.model.tokenizer))])
        return phrase_reps, phrase_sources
    
    @torch.no_grad()
    def get_prefix_phrases_fast(self, documents):
        self.model.eval()
        batch = self.model.bert_tokenizer.batch_encode_plus(documents, padding=True, return_tensors='pt', max_length=200, truncation=True)
        input_ids = batch['input_ids'].cuda()
        mask = batch['attention_mask'].cuda()

        # replace the [CLS] with [PREFIX] for the prefix text (document)
        input_ids[0, 0] = self.model.prefix_token_id

        vl = mask.sum(dim=-1)
        hidden_states = self.model.phrase_encoder(input_ids, mask, output_hidden_states=True)['hidden_states'][-1]    # [B, S, E]

        begin_rep, end_rep = [], []
        phrase_sources = []
        input_ids = input_ids.tolist()
        for doc_rep, l, doc_id in zip(hidden_states, vl, input_ids):
            s_pos, e_pos = [], []
            for i in range(1, l-self.args['left_window_size']):
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    s_pos.append(i)
                    e_pos.append(j)
                    phrase_sources.append(doc_id[i:j+1])
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.model.s_proj(begin_rep), self.model.e_proj(end_rep)], dim=-1)
        return phrase_reps, phrase_sources

    def train_model_gpt2(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        with autocast():
            batch['current_step'] = current_step
            loss, acc = self.model(batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

        if recoder:
            recoder.add_scalar(f'train/RunLoss', loss.item(), current_step)
            recoder.add_scalar(f'train/Tokenacc', acc, current_step)
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; acc: {round(acc, 4)}')
        pbar.update(1)

    @torch.no_grad()
    def gpt2_generation(self, prefix, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        # maximum 128 tokens
        input_ids = self.model.vocab.encode(prefix, add_special_tokens=False)
        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).cuda()
        length = len(input_ids[0])
        use_cache = False if get_time_cost else True
        bt = time.time()
        if decoding_method == 'nucleus_sampling':
            output = self.model.model.generate(
                input_ids,
                do_sample=True,
                max_length=length+128,
                top_p=top_p,
                top_k=0,
                use_cache=use_cache
            )
        else:
            output = self.model.model.generate(
                input_ids,
                max_length=length+128,
                use_cache=use_cache
            )
        inference_time = time.time() - bt
        string = self.model.vocab.decode(output[0, length:])
        return string, inference_time

    @torch.no_grad()
    def knnlm_generation(self, prefix, decoding_method='nucleus_sampling', top_k=0, top_p=0.95, temp=1.0, get_time_cost=False):
        # maximum 128 tokens
        input_ids = self.model.vocab.encode(prefix, add_special_tokens=False)
        input_ids = torch.LongTensor(input_ids).unsqueeze(dim=0).cuda()
        length = len(input_ids[0])
        bt = time.time()
        if decoding_method == 'nucleus_sampling':
            string = self.model.nucleus_sampling(
                input_ids,
                max_length=128,
                top_p=top_p,
            )
        elif decoding_method == 'greedy':
            string = self.model.greedy_search(
                input_ids,
                max_length=128,
            )
        return string, time.time() - bt

    @torch.no_grad()
    def inference_knnlm(self, inf_iter, size=500000):
        self.model.eval()
        embds, texts = [], []
        counter = 0
        for batch in tqdm(inf_iter):
            rep, target = self.model(batch)
            embds.append(rep)
            texts.extend(target)
            if len(texts) > size:
                embds = torch.cat(embds, dim=0).numpy()
                torch.save(
                    (embds, texts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}_1024/knnlm/inference_{self.args["local_rank"]}_{counter}.pt'
                )
                counter += 1
                texts, embds = [], []
        if len(texts) > 0:
            embds = torch.cat(embds, dim=0).numpy()
            torch.save(
                (embds, texts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}_1024/knnlm/inference_{self.args["local_rank"]}_{counter}.pt'
            )

    @torch.no_grad()
    def test_model_ppl(self, test_iter, max_counter=10000):
        ppls = []
        counter = 0
        pbar = tqdm(test_iter)
        for batch in pbar:
            ppl = self.model.calculate_ppl(batch)
            ppls.append(ppl)
            counter += 1
            if counter >= max_counter:
                break
            ppl = np.mean(ppls)
            pbar.set_description(f'[!] ppl: {round(ppl, 4)}')
        print('Perplexity:', round(ppl, 4))


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits



