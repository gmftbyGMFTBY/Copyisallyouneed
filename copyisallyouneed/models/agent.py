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
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            try:
                self.model.module.load_state_dict(state_dict)
            except:
                self.model.load_state_dict(state_dict)
        print(f'[!] load model from {path}')
    
    def train_model(self, batch, recoder=None, current_step=0, pbar=None):
        self.model.train()
        with autocast():
            batch['current_step'] = current_step
            loss_1, loss_2, loss_3, loss_4, acc_1, acc_2, acc_3, acc_4 = self.model(batch)
            loss = loss_1 + loss_2 + loss_3 + loss_4
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
            # recoder.add_scalar(f'train/pure_token_head_loss', loss_0.item(), current_step)
            recoder.add_scalar(f'train/token_head_loss', loss_1.item(), current_step)
            recoder.add_scalar(f'train/token_tail_loss', loss_2.item(), current_step)
            recoder.add_scalar(f'train/phrase_head_loss', loss_3.item(), current_step)
            recoder.add_scalar(f'train/phrase_tail_loss', loss_4.item(), current_step)
            # recoder.add_scalar(f'train/pure_token_acc', acc_0, current_step)
            recoder.add_scalar(f'train/token_head_acc', acc_1, current_step)
            recoder.add_scalar(f'train/token_tail_acc', acc_2, current_step)
            recoder.add_scalar(f'train/phrase_head_acc', acc_3, current_step)
            recoder.add_scalar(f'train/phrase_tail_acc', acc_4, current_step)
        pbar.set_description(f'[!] loss: {round(loss_1.item(), 4)}|{round(loss_2.item(), 4)}|{round(loss_3.item(), 4)}|{round(loss_4.item(), 4)}; acc: {round(acc_1, 4)}|{round(acc_2, 4)}|{round(acc_3, 4)}|{round(acc_4, 4)}')
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

