import os

import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

from model.score_net import EDM_Net
from sampler import edm_sampler, naive_sampler
from dataloader import data_rescale, inverse_data_rescale
from utils.misc import is_master, load_scorenet
from .base_trainer import Base_Trainer
from .loss import Loss_EDM
from .ema import EMAHelper

"""
TODO:
- setup the training for img data
    - Unet √
    - multi-gpu √
    - ema √
    - snapshot and continue training √
- evaluation
    - test FID √
    - evaluate likelihood
- misc
    - add label and conditional
- distillation
    - 1d toy data with trajectory plot
    - flow matching
"""

class Score_Trainer(Base_Trainer):
    def __init__(self, config, logger, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, logger, log_dir, ckpt_dir, sample_dir)
        self._build_scorenet()
        self._build_optimizer()
        self.loss_fn = Loss_EDM()
        self.net = DDP(self.net, device_ids=[self.device])
        if is_master() and self.config['training']['ema']:
            self.ema_helper = EMAHelper(self.config['training']['ema_rate'])
            self.ema_helper.register(self.net.module)

    def _build_scorenet(self):
        if self.config['training']['resume']:
            self.net, history_iters = load_scorenet(self.config['training']['resume_path'])
            self.history_iters = history_iters
            self.net = self.net.to(self.device)
            if is_master():
                self.logger.info(f'Resume training, history_iters: {self.history_iters}')
        else:
            self.net = EDM_Net(**self.config['score_net']).to(self.device)
            if is_master():
                self.logger.info('Train from scratch')

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
                    loss = self.loss_fn(net=self.net, x=x)
                    if is_master():
                        self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1
                    self.history_iters += 1
                    if is_master() and self.config['training']['ema']:
                        self.ema_helper.update(self.net.module)

                    if self.num_iters % int(self.config['training']['sample_iters']) == 0:
                        if is_master():
                            if self.config['data']['type'] == '2d':
                                self.sample_2d(num_steps=500)
                                self.logger.debug('sample')
                            else:
                                samples = self.sample_img_rgb(img_size=self.config['data']['img_size'], num_samples=64, num_steps=100)
                                if self.config['data']['rescale']:
                                    samples = inverse_data_rescale(samples)
                                self.writer.add_images('samples', samples, self.num_iters)
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
    def sample_2d(self, num_steps=1000, sampler=edm_sampler):
        self.net.eval()
        latents = torch.randn((5000, 2)).to(self.device)
        net = self.ema_helper.shadow if self.config['training']['ema'] else self.net.module
        samples = sampler(net, latents, verbose=False, num_steps=num_steps)
        samples = samples.detach().cpu()
        plt.scatter(samples[:, 0], samples[:, 1], s=1)
        save_path = os.path.join(self.sample_dir, 'sample_'+str(self.num_iters)+'.png')
        plt.savefig(save_path)
        plt.close()
        self.net.train()
    
    @torch.no_grad()
    def sample_img_rgb(self, img_size, num_samples=16, num_steps=1000, sampler=edm_sampler):
        self.net.eval()
        latents = torch.randn((num_samples, 3, img_size, img_size)).to(self.device)
        net = self.ema_helper.shadow if self.config['training']['ema'] else self.net.module
        samples = sampler(net, latents, verbose=False, num_steps=num_steps)
        samples = samples.detach().cpu()
        self.net.train()
        return samples

    # @staticmethod
    # def denormalize(image):
    #     return image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)

    