from header import *

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
        self.load_latest_checkpoint()

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
    def generate_one_sample(self, text, retriever, decoding_method='greedy', top_k=0, top_p=0.95, temp=1.0):
        self.model.eval()
        ids = self.model.tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'].cuda()
        prefix_length = len(ids[0])
        # retrieve and encode the documents
        documents = retriever.search([text], self.args['doc_topk'])[0]
        # add the prefix
        documents = [text] + documents
        phrase_reps, phrase_sources = self.get_phrases_fast(documents)
        candidates = []
        while len(ids[0]) <= prefix_length + self.args['max_gen_len']:
            ids, candidate = self.generate_one_step_fast(ids, phrase_reps, phrase_sources, decoding_method=decoding_method, top_k=top_k, top_p=top_p, temp=temp)
            candidates.append(candidate)
        return self.model.tokenizer.decode(ids[0, prefix_length:]), candidates

    @torch.no_grad()
    def generate_one_step_fast(self, ids, phrase_reps, phrase_sources, decoding_method='greedy', temp=1., top_k=0, top_p=0.92):
        self.model.eval()
        query = self.model.get_query_rep(ids)
        candidates = self.search_from_documents_fast(query, phrase_reps, phrase_sources, search_topk=self.args['phrase_topk'])
        candidates = sorted(candidates, key=lambda x:x[1], reverse=True)

        if decoding_method == 'greedy':
            candidate, _ = candidates[0]
        elif decoding_method == 'nucleus_sampling':
            scores = torch.tensor([j for i, j in candidates])
            scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            index = torch.multinomial(F.softmax(scores / temp, dim=-1), num_samples=1).item()
            candidate, _ = candidates[index]
        else:
            pass

        # get textual candidate
        if type(candidate) == torch.Tensor:
            candidate = ' ' + self.model.bert_tokenizer.decode(candidate).replace('[UNK]', '<|endoftext|>')
            sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
        else:
            sub_ids = [candidate]
        sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        ids = torch.cat((ids, sub_ids), dim=-1)
        return ids, candidate

    @torch.no_grad()
    def generate_one_step(self, ids, phrase_reps, phrase_sources, decoding_method='greedy', temp=1., top_k=0, top_p=0.92):
        self.model.eval()
        query = self.model.get_query_rep(ids)
        candidates = self.search_from_documents(query, phrase_reps, phrase_sources, search_topk=self.args['phrase_topk'])
        candidates = [i for i in candidates if '<|endoftext|>' not in i[0]]
        candidates = sorted(candidates, key=lambda x:x[1], reverse=True)

        if decoding_method == 'greedy':
            candidate, _ = candidates[0]
        elif decoding_method == 'nucleus_sampling':
            scores = torch.tensor([j for i, j in candidates])
            scores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            index = torch.multinomial(F.softmax(scores / temp, dim=-1), num_samples=1).item()
            candidate, _ = candidates[index]
        else:
            pass
        sub_ids = self.model.tokenizer.encode(candidate, add_special_tokens=False)
        sub_ids = torch.LongTensor(sub_ids).unsqueeze(0).cuda()
        ids = torch.cat((ids, sub_ids), dim=-1)
        return ids, candidate

    @torch.no_grad()
    def get_phrases(self, documents):
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
        for doc_rep, l, doc_id in tqdm(zip(hidden_states, vl, input_ids)):
            s_pos, e_pos = [], []
            for i in range(1, l-self.args['left_window_size']):
                for j in range(
                    min(i+self.args['left_window_size'], l-1), 
                    min(i+self.args['right_window_size'], l-1)
                ):
                    string = self.model.bert_tokenizer.decode(doc_id[j])
                    
                    if string.startswith('##'):
                        # remove the last and append the new
                        if s_pos and e_pos:
                            s_pos.pop()
                            e_pos.pop()
                    s_pos.append(i)
                    e_pos.append(j)
                last_index = min(i+self.args['right_window_size'], l-1)
                string = self.model.bert_tokenizer.decode(doc_id[last_index])
                if string.startswith('##'):
                    if s_pos and e_pos:
                       s_pos.pop()
                       e_pos.pop()
            for s, e in zip(s_pos, e_pos):
                string = ' ' + self.model.bert_tokenizer.decode(doc_id[s:e+1]).replace('[UNK]', '<|endoftext|>')
                phrase_sources.append((s, e, string))
            s_rep = doc_rep[s_pos, :]
            e_rep = doc_rep[e_pos, :]
            begin_rep.append(s_rep)
            end_rep.append(e_rep)
        begin_rep = torch.cat(begin_rep)
        end_rep = torch.cat(end_rep)
        phrase_reps = torch.cat([self.model.s_proj(begin_rep), self.model.e_proj(end_rep)], dim=-1)
        phrase_reps = torch.cat([
            phrase_reps,
            self.model.token_embeddings
        ], dim=0)
        phrase_sources.extend([
            (
                -1, 
                -1, 
                ' ' + self.model.tokenizer.decode(idx) if self.model.tokenizer.decode(idx) in ['.', ',', '!', ';', ':', '"', "'", '?', '#', '$', '%', '/', '&', '*', '(', ')', '[', ']', '{', '}', '|'] else self.model.tokenizer.decode(idx),
            ) for idx in range(len(self.model.tokenizer))
        ])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources

    def search_from_documents(self, query, phrase_reps, phrase_source, search_topk=5):
        dp = torch.matmul(query, phrase_reps.t()).squeeze(0)   
        search_num = min(search_topk, len(phrase_reps))
        dis, topk = dp.topk(search_num, dim=-1)    # [K]
        dis = dis.tolist()
        candidates = [(phrase_source[i][-1], round(d, 4)) for i, d in zip(topk, dis)]
        return candidates

    def search_from_documents_fast(self, query, phrase_reps, phrase_source, search_topk=5):
        dp = torch.matmul(query, phrase_reps.t()).squeeze(0)   
        search_num = min(search_topk, len(phrase_reps))
        dis, topk = dp.topk(search_num, dim=-1)    # [K]
        dis = dis.tolist()
        candidates = [(phrase_source[i], round(d, 4)) for i, d in zip(topk, dis)]
        return candidates

    @torch.no_grad()
    def get_phrases_fast(self, documents):
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
        for doc_rep, l, doc_id in tqdm(zip(hidden_states, vl, input_ids)):
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
        phrase_reps = torch.cat([
            phrase_reps,
            self.model.token_embeddings
        ], dim=0)
        phrase_sources.extend([idx for idx in range(len(self.model.tokenizer))])
        print(f'[!] add vocabulary and collect {len(phrase_reps)} phrases')
        return phrase_reps, phrase_sources


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


