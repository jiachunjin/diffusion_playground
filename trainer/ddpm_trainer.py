import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, SGD
from tqdm.autonotebook import tqdm, trange

from model.diffusion import Diffusion
from model.unet import UNet
from dataloader import prepare_dataloader
from utils import save_img_tensors


class DDPM_Trainer():
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.diffusion = Diffusion(self.config['diffusion_config'])
        if config['parametric']:
            self._build_model_parametric()
        else:
            self._build_model_nonparametric()
        self._build_dataloader()
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.num_iters = 0
        self.start_iter = 0

    def _build_model_nonparametric(self):
        """
        Particle based score matching
        """
        K = self.config['K']
        num_channel = self.config['data_config']['num_channel']
        img_size = self.config['data_config']['img_size']
        lr = self.config['optimizer_config']['lr']
        self.X_hat = 0.5 * torch.ones(self.config['diffusion_config']['timesteps'], K, num_channel, img_size, img_size).to(self.device)
        self.X_hat.requires_grad = True
        self.optimizer = SGD(
            [
                {'params': self.X_hat},
            ],
            lr = float(lr),
        )

    def _build_model_parametric(self):
        #TODO Use U-Net to build the score model
        lr = self.config['optimizer_config']['lr']
        self.unet = UNet(**self.config['unet_config']).to(self.device)
        self.optimizer = Adam(
            [
                {'params': self.unet.parameters()},
            ],
            lr = float(lr),
        )

    def _build_dataloader(self):
        root = self.config['data_config']['root']
        name = self.config['data_config']['name']
        train_batch = self.config['trainer_config']['batch_size']
        test_batch = self.config['trainer_config']['batch_size']
        num_workers = self.config['trainer_config']['num_workers']

        self.train_loader, self.test_loader = prepare_dataloader(root, name, train_batch, test_batch, num_workers)
    
    def train(self):
        total_iters = int(self.config['trainer_config']['total_iters'])
        sample_every_iters = int(self.config['trainer_config']['sample_every_iters'])
        save_every_iters = int(self.config['trainer_config']['save_every_iters'])
        done = False
        with tqdm(total=total_iters) as pbar:
            pbar.update(self.num_iters)
            while not done:
                for x, _ in self.train_loader:
                    x = x.to(self.device)
                    self.optimizer.zero_grad()
                    if self.config['parametric']:
                        loss = self.diffusion.score_matching_loss_parametric(self.unet, x)
                    else:
                        loss = self.diffusion.score_matching_loss_nonparametric(self.X_hat, x)
                    self.writer.add_scalar('loss', loss.item(), self.num_iters)
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                    self.num_iters += 1

                    if self.num_iters % sample_every_iters == 0:
                        # sample here
                        if self.config['parametric']:
                            samples = self.diffusion.sample_parametric(self.unet, 32, self.config['data_config']['img_size'])
                            self.writer.add_images('samples', self.denormalize(samples.detach()), self.num_iters)
                        else:
                            self.writer.add_images('X_hat', self.denormalize(self.X_hat.detach()[:64]), self.num_iters)
                            samples = self.diffusion.sample_nonparametric(self.X_hat, 64)
                            self.writer.add_images('samples', self.denormalize(samples.detach()), self.num_iters)
                    if self.num_iters % save_every_iters == 0:
                        # save here
                        if self.config['parametric']:
                            self.save()
                        else:
                            raise NotImplementedError


                    if self.num_iters >= total_iters:
                        done = True
                        break

    def save(self):
        model_path = os.path.join(self.ckpt_dir, f'model_{self.num_iters}.pt')
        data = {
            'unet': self.unet.state_dict(),
            'num_iters': self.num_iters,
        }
        torch.save(data, model_path)

    @staticmethod
    def denormalize(image):
        return image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).to('cpu', torch.uint8)