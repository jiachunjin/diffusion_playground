import torch


class Loss_EDM():
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x):
        """
        y = x + n, where x is the clean signal
        """
        if x.dim() == 2:
            # toy data
            rnd_normal = torch.randn([x.shape[0], 1], device=x.device)
        elif x.dim() == 4:
            # image data
            rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        D_xn = net(x + n, sigma)
        loss = weight * ((D_xn - x) ** 2)
        return loss.mean()