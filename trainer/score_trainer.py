import os
import pickle
import copy

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from model.score_net import EDM_Net
from sampler import edm_sampler, naive_sampler
from .base_trainer import Base_Trainer
from .loss import Loss_EDM

"""
TODO:
- setup the training for img data
    - Unet √
    - ema
    - multi-gpu
- add label and conditional
"""

class Score_Trainer(Base_Trainer):
    def __init__(self, config, logger, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, logger, log_dir, ckpt_dir, sample_dir)
        self._build_scorenet()
        self._build_optimizer()
        self.loss_fn = Loss_EDM()

    def _build_scorenet(self):
        self.net = EDM_Net(**self.config['score_net']).to(self.device)
        self.logger.info('Network built')

    def train(self):
        total_iters = int(self.config['training']['iters'])
        done = False
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                for x, y in self.train_loader:
                    x = x.float().to(self.device)
                    loss = self.loss_fn(net=self.net, x=x)
                    self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1

                    if self.num_iters % int(self.config['training']['sample_iters']) == 0:
                        if self.config['data']['type'] == '2d':
                            self.sample_2d(num_steps=500)
                            self.logger.debug('sample')
                        else:
                            samples = self.sample_img_rgb(img_size=self.config['data']['img_size'], num_samples=64, num_steps=100)
                            self.writer.add_images('samples', samples, self.num_iters)
                    if self.num_iters % int(self.config['training']['save_iters']) == 0:
                        self.save()
                        self.logger.debug('Saved')

                    if self.num_iters >= total_iters:
                        done = True
                        break

    def save(self):
        state = {
            'net': self.net.state_dict(),
            'net_config': self.config['score_net']
        }
        save_path = os.path.join(self.ckpt_dir, f'{self.config["data"]["name"]}-{self.num_iters//1000:06d}.pt')
        torch.save(state, save_path)

    @torch.no_grad()
    def sample_2d(self, num_steps=1000, sampler=edm_sampler):
        self.net.eval()
        latents = torch.randn((5000, 2)).to(self.device)
        samples = sampler(self.net, latents, verbose=False, num_steps=num_steps)
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
        samples = sampler(self.net, latents, verbose=False, num_steps=num_steps)
        samples = samples.detach().cpu()
        self.net.train()
        return samples

    # @staticmethod
    # def denormalize(image):
    #     return image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)

    