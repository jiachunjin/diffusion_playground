import os
import torch
import matplotlib.pyplot as plt

from tqdm.autonotebook import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from .base_trainer import Base_Trainer
from .ema import EMAHelper
from .loss import Loss_CD

from dataloader import data_rescale, inverse_data_rescale
from model.score_net import EDM_Net
from utils.misc import load_scorenet, requires_grad, is_master, append_dims

"""
TODO:
- finish the consistency loss
    - can run on 2d data loss.py
    - support different data dimensions (append_dims() in original CM code)

"""


class CD_Trainer(Base_Trainer):
    def __init__(self, config, logger, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, logger, log_dir, ckpt_dir, sample_dir)
        self._build_scorenet()
        self._build_optimizer()
        self.loss_fn = Loss_CD(device=self.device)
        self.net = DDP(self.net, device_ids=[self.device])
        self.ema_helper = EMAHelper(self.config['training']['ema_rate'])
        # self.ema_helper = EMAHelper(0)
        self.ema_helper.register(self.net.module)

    def _build_scorenet(self):
        teacher_path = self.config['training']['teacher_path']
        self.teacher, history_iters = load_scorenet(teacher_path)
        self.teacher = self.teacher.to(self.device)
        # self.net = EDM_Net(**self.config['score_net']).to(self.device)
        self.net, _ = load_scorenet(teacher_path)
        self.net = self.net.to(self.device)
        self.teacher.eval()
        requires_grad(self.teacher.parameters(), False)
        self.logger.info(f'Load teacher model from {teacher_path}, which is trained for {history_iters} iters.')
    
    def train(self):
        total_iters = int(self.config['training']['iters'])
        done = False
        epoch = 0
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                if torch.cuda.device_count() > 1:
                    self.train_loader.sampler.set_epoch(epoch)
                for x, y in self.train_loader:
                    x = x.float().to(self.device)
                    if self.config['data']['rescale']:
                        x = data_rescale(x)
                    loss = self.loss_fn(online_net=self.net.module, target_net=self.ema_helper.shadow, teacher=self.teacher, x=x)
                    if is_master():
                        self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1
                    self.history_iters += 1
                    self.ema_helper.update(self.net.module)

                    if self.num_iters % int(self.config['training']['sample_iters']) == 0:
                        if is_master():
                            if self.config['data']['type'] == '2d':
                                self.sample_2d()
                                self.logger.debug('sample')
                            else:
                                raise NotImplementedError
                    
                    if self.num_iters % int(self.config['training']['save_iters']) == 0:
                        if is_master():
                            self.save()
                            self.logger.debug(f'Saved at history_iters: {self.history_iters}')

                    if self.num_iters >= total_iters:
                        done = True
                        break
                epoch += 1

    def save(self):
        state = {
            'net': self.net.module.state_dict(),
            'ema': self.ema_helper.shadow.state_dict(),
            'net_config': self.config['score_net'],
            'history_iters': self.history_iters,
        }
        save_path = os.path.join(self.ckpt_dir, f'{self.config["data"]["name"]}.pt')
        torch.save(state, save_path)

    @torch.no_grad()
    def sample_2d(self):
        latents = self.loss_fn.sigma_max * torch.randn((50000, 2)).to(self.device)
        net = self.ema_helper.shadow
        # net = self.net.module
        net.eval()
        samples = net(latents, append_dims(4*torch.ones((50000,), dtype=torch.float32, device=latents.device), latents.ndim))
        samples = samples.detach().cpu()
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        # plt.xlim(-5, 5)
        # plt.ylim(-5, 5)
        self.writer.add_figure('samples', plt.gcf(), self.num_iters)
        # save_path = os.path.join(self.sample_dir, 'sample_'+str(self.num_iters)+'.png')
        # plt.savefig(save_path)
        # plt.close()


