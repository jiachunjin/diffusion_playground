import os
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm, trange

from .base_trainer import Base_Trainer
from .loss import Loss_EDM
from utils.visualize import plot_tensors

"""
TODO:
- ema
- setup the training
- make the net consistent
- add label and conditional
- multi-gpu
"""

class Score_Trainer(Base_Trainer):
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        super().__init__(config, log_dir, ckpt_dir, sample_dir)
        self._build_scorenet()
        self._build_optimizer()
        self.loss_fn = Loss_EDM()

    def train(self):
        self.net.train()
        total_iters = int(self.config['training']['iters'])
        done = False
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                for x in self.train_loader:
                    x = x.float().to(self.device)
                    loss = self.loss_fn(net=self.net, x=x)
                    self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    # logging.info(f'step: {self.num_iters}, loss: {loss.item()}')
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1

                    if self.num_iters % int(self.config['training']['sample_iters']) == 0:
                        # sample here
                        latents = torch.randn((4000, 2))
                        samples = edm_sampler(self.net, latents, num_steps=200)
                        samples = samples.detach()
                        plt.scatter(samples[:, 0], samples[:, 1], s=1)
                        save_path = os.path.join(self.sample_dir, 'sample_'+str(self.num_iters)+'.png')
                        plt.savefig(save_path)
                        plt.close()
                        # self.writer.add_images('samples', self.denormalize(samples.detach()), self.num_iters)
                    #     else:
                    #         self.writer.add_images('X_hat', self.denormalize(self.X_hat.detach()[:64]), self.num_iters)
                    #         samples = self.diffusion.sample_nonparametric(self.X_hat, 64)
                    #         self.writer.add_images('samples', self.denormalize(samples.detach()), self.num_iters)
                    # if self.num_iters % int(self.config['training']['save_iters']) == 0:
                    #     # save here
                    #     if self.config['parametric']:
                    #         self.save()
                    #     else:
                    #         raise NotImplementedError

                    if self.num_iters >= total_iters:
                        done = True
                        break

    # def save(self):
    #     model_path = os.path.join(self.ckpt_dir, f'model_{self.num_iters}.pt')
    #     data = {
    #         'unet': self.unet.state_dict(),
    #         'num_iters': self.num_iters,
    #     }
    #     torch.save(data, model_path)

    # @staticmethod
    # def denormalize(image):
    #     return image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)


def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float32) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next