import torch
from utils.misc import append_dims

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


class Loss_CD():
    """
    Consistency distillation loss
    """
    def __init__(self, device, num_steps=20, sigma_max=8, rho=7, sigma_min=0.002):
        self.num_steps = num_steps
        self.sigma_max = sigma_max # sigma_max=80 is too large for 2d toy data
        self.sigma_min = sigma_min
        self.rho = rho
        self.device = device
        # Note that the step_indices is different from the one in EDM
        step_indices = torch.arange(self.num_steps, dtype=torch.float32, device=device)
        t_steps = (sigma_min ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_max ** (1 / rho) - sigma_min ** (1 / rho))) ** rho
        self.t_steps = torch.as_tensor(t_steps)

    def __call__(self, online_net, target_net, teacher, x):
        online_net.train()
        target_net.eval()
        n = torch.randint(0, self.num_steps - 1, (x.shape[0],), device=self.device, dtype=torch.long)
        t_cur = self.t_steps[n]
        t_next = self.t_steps[n+1]
        x_next = x + append_dims(t_next, x.ndim) * torch.randn_like(x, device=x.device)
        output_online = online_net(x_next, append_dims(t_next, x_next.ndim))
        # x_cur = self.euler_solver(x_next, teacher, t_next, t_cur).detach()
        x_cur = self.heun_solver(x_next, teacher, t_next, t_cur).detach()
        output_target = target_net(x_cur, append_dims(t_cur, x_cur.ndim)).detach()
        loss = self.l2_loss(output_online, output_target).mean()
        return loss
        
    @torch.no_grad()
    def euler_solver(self, x_next, teacher, t_next, t_cur):
        teacher.eval()
        d = (x_next - teacher(x_next, append_dims(t_next, x_next.ndim)).to(torch.float32)) / append_dims(t_next, x_next.ndim)
        x_cur = x_next + append_dims(t_cur - t_next, x_next.ndim) * d
        return x_cur
    
    @torch.no_grad()
    def heun_solver(self, x_next, teacher, t_next, t_cur):
        """
        t_next: t1
        t_cur: t2
        x_next: x_1
        """
        teacher.eval()
        denoised = teacher(x_next, append_dims(t_next, x_next.ndim)).to(torch.float32)
        d = (x_next - denoised) / append_dims(t_next, x_next.ndim)

        sample_temp = x_next + d * append_dims(t_cur - t_next, x_next.ndim)
        denoised_2 = teacher(sample_temp, append_dims(t_cur, sample_temp.ndim))
        d_2 = (sample_temp - denoised_2) / append_dims(t_cur, x_next.ndim)
        d_prime = (d + d_2) / 2
        samples = x_next + d_prime * append_dims(t_cur - t_next, x_next.ndim)

        return samples






    def l2_loss(self, output_online, output_target):
        diff = (output_online - output_target) ** 2
        return diff.sum(dim=[i for i in range(1, diff.ndim)])

