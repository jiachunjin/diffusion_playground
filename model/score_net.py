import torch
import numpy as np
import logging

from .module import (
    Linear
)


class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        modules = []
        dim_input = dim + 1 # concatenated with the scalar sigma
        for i in hidden_dim:
            modules.extend([
                Linear(dim_input, i),
                torch.nn.SiLU(),
                torch.nn.Dropout(dropout_p)
            ])
            dim_input = i
        modules.append(torch.nn.Linear(dim_input, dim))
        self.net = torch.nn.Sequential(*modules)
    
    def forward(self, x, noise_labels):
        if not noise_labels.dim():
            noise_labels = noise_labels.expand([x.shape[0], 1])
        inputs = torch.cat([x, noise_labels], dim=1)
        return self.net(inputs)


class EDM_Net(torch.nn.Module):
    def __init__(self, model_type, sigma_min=0, sigma_max=float('inf'), sigma_data=0.5, **kwargs):
        super().__init__()
        if model_type == 'MLP':
            self.net = MLP(kwargs['dim'], kwargs['hidden_dim'], kwargs['dropout_p'])
        elif model_type == 'Unet':
            raise NotImplementedError
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.net((c_in * x), c_noise)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
