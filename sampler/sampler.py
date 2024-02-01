import torch
import numpy as np
from tqdm.autonotebook import tqdm

def naive_sampler(net, latents, verbose=True, num_steps=200, rho=7, sigma_max=80, sigma_min=0.002):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    x_next = latents.to(torch.float32) * t_steps[0]
    with tqdm(total=num_steps, disable=(not verbose)) as pbar:
        pbar.update(0)
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            if latents.dim() == 4:
                B = latents.shape[0]
                t_cur = t_cur.repeat(B).view(-1, 1, 1, 1)
            x_cur = x_next
            d = (x_cur - net(x_cur, t_cur).to(torch.float32)) / t_cur
            x_next = x_cur + (t_next - t_cur) * d
            pbar.update(1)
    return x_next



def edm_sampler(
    net, latents, verbose=True, class_labels=None, randn_like=torch.randn_like,
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
    with tqdm(total=num_steps, disable=(not verbose)) as pbar:
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            if latents.dim() == 4:
                B = latents.shape[0]
                t_hat = t_hat.repeat(B).view(-1, 1, 1, 1)
                t_next = t_next.repeat(B).view(-1, 1, 1, 1)
            denoised = net(x_hat, t_hat).to(torch.float32)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = net(x_next, t_next).to(torch.float32)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            pbar.update(1)
    return x_next