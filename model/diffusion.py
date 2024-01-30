import torch
import numpy as np
from functools import partial
from tqdm.autonotebook import trange
import warnings
warnings.filterwarnings('ignore')

class Diffusion():
    def __init__(self, config):
        self.device = config['device']
        self.timesteps = config['timesteps']
        betas = np.linspace(config['linear_beta_start'], config['linear_beta_end'], self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        self.to_torch = to_torch

        self.alphas = to_torch(alphas)
        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        # self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)
        # self.alphas_cumprod_next = to_torch(alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        # self.sqrt_recip_alphas_cumprod_m1 = to_torch(np.sqrt(1. / alphas_cumprod - 1.))

    def score_matching_loss_nonparametric(self, X_hat, x):
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x_0=x, t=t, noise=noise)
        true_score = self.get_true_score(noise, t, x.shape)
        t_ = self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, x.shape)
        return torch.mean((self.score_nonparametric(x_t, t_, X_hat) - true_score) ** 2)

    def score_nonparametric(self, z, t, X_hat):
        B, C, img_size, _ = z.shape
        K = X_hat.shape[0]
        t_expanded = t.view(B, 1, 1, 1, 1).expand(B, K, C, img_size, img_size)
        X_hat_expanded = X_hat.unsqueeze(dim=0)
        z_expanded = z.unsqueeze(dim=1)
        nom = -((z_expanded - X_hat_expanded * t_expanded)**2).sum(dim=[i for i in range(2, z.dim()+1)])
        den = (2 * (1 - t**2)).view(-1, 1)
        w_ = nom / den
        w = torch.softmax(w_, dim=1).view(B, K, 1, 1, 1)
        weighted_sum = torch.sum(X_hat_expanded * w, dim=1)
        return (weighted_sum * t.view(-1, 1, 1, 1) - z) / (1 - t**2).view(-1, 1, 1, 1)

    def sample_nonparametric(self, X_hat, num_samples):
        X_hat = X_hat.detach()
        K, C, img_size, img_size = X_hat.shape
        z = torch.randn(num_samples, 1, img_size, img_size).to(self.device)
        for t in trange(len(self.betas)-1, 0, -1, leave=False):
            beta = self.betas[t]
            s = self.score_nonparametric(z, torch.tensor(self.sqrt_alphas_cumprod[t]).expand(num_samples,), X_hat)
            z = (z + beta * s) / self.alphas[t] + beta.sqrt() * torch.randn_like(z, device=self.device)
        return z.detach().cpu()

    def q_sample(self, x_0, t, noise):
        shape = x_0.shape
        return (
            self.extract_coef_at_t(self.sqrt_alphas_cumprod, t, shape) * x_0
            + self.extract_coef_at_t(self.sqrt_one_minus_alphas_cumprod, t, shape) * noise
        )
    
    def get_true_score(self, noise, t, shape):
        return -self.extract_coef_at_t(self.sqrt_recip_alphas_cumprod, t, shape) * noise
    
    def score_matching_loss_parametric(self, unet, x):
        batch_size = x.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = self.q_sample(x_0=x, t=t, noise=noise)
        true_score = self.get_true_score(noise, t, x.shape)
        return torch.mean((unet(x_t, t) - true_score) ** 2)

    @torch.no_grad()
    def sample_parametric(self, unet, num_samples, img_size):
        z = torch.randn(num_samples, 1, img_size, img_size).to(self.device)
        for t in trange(len(self.betas)-1, 0, -1, leave=False):
            beta = self.betas[t]
            s = unet(z, t * torch.ones((num_samples, ), device=self.device, dtype=torch.long))
            z = (z + beta * s) / self.alphas[t] + beta.sqrt() * torch.randn_like(z, device=self.device)
        return z.detach().cpu()

    @staticmethod
    def extract_coef_at_t(schedule, t, x_shape):
        return torch.gather(schedule, -1, t).reshape([x_shape[0]] + [1] * (len(x_shape) - 1))