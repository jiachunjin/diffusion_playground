import os
import glob
import argparse

import torch
from cleanfid import fid
from tqdm.autonotebook import trange
import torchvision.utils as tvu

from sampler import edm_sampler, naive_sampler
from dataloader import inverse_data_rescale
from utils.misc import load_scorenet



def sample_fid(net, sampler, num_steps, batch_shape, num_samples, output_path):
    if not os.path.exists(output_path): 
        os.makedirs(output_path)
    else:
        files = glob.glob(os.path.join(output_path, '*'))
        for f in files:
            os.remove(f)
    num_rounds = num_samples // batch_shape[0]
    img_id = 0
    with torch.no_grad():
        for i in trange(num_rounds):
            z = torch.randn(batch_shape).cuda()
            if sampler == 'edm_sampler':
                samples = edm_sampler(net, z, num_steps=num_steps, verbose=True)
            elif sampler == 'naive_sampler':    
                samples = naive_sampler(net, z, num_steps=num_steps, verbose=True)
            samples = inverse_data_rescale(samples.detach().cpu())
            for j in range(batch_shape[0]):
                path = os.path.join(output_path, f"{img_id}.png")
                tvu.save_image(samples[j], path)
                img_id += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--ckpt_path', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, choices=['cifar10'], default='cifar10')
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('-t', '--num_steps', type=int, default=100)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-s', '--sampler', type=str, choices=['edm_sampler', 'naive_sampler'], default='edm_sampler')
    args = parser.parse_args()
    if args.num_samples % args.batch_size != 0:
        raise ValueError('num_samples for sampling must be divided exactly batch_size.')

    if args.data == 'cifar10':
        batch_shape = (args.batch_size, 3, 32, 32)

    net, history_iters = load_scorenet(args.ckpt_path, use_ema=True)
    print(f'history_iters: {history_iters}')
    net = net.cuda()
    sample_fid(net, args.sampler, args.num_steps, batch_shape, args.num_samples, args.output_path)
    print(f'sample fid done, saved at {args.output_path}')
    score = fid.compute_fid(args.output_path, dataset_name='cifar10', dataset_res=32, dataset_split='train')
    print(f'score: {score}')





"""
    CUDA_VISIBLE_DEVICES="5" python fid.py -p experiments/cifar_new_240208-10:03:27/checkpoints/cifar10.pt -b 1000 -n 5000 -t 50 -o fid_samples/tem
    50000: 27.443
    55000: 23.861
    60000: 21.716
    65000: 19.939
    70000: 19.666
    75000: 18.743
    80000: 17.810
    85000: 17.035
    100000: 16.439
    110000: 15.501
    120000: 15.247
    135000: 14.518
    310000: 11.579
    330000: 11.984
    350000: 11.765
    370000: 11.746
    380000: 11.439 **
    400000: 12.733
    410000: 11.925
    430000: 11.812
    440000: 12.132
    450000: 11.852
    460000: 11.794
    590000: 12.191
    610000: 12.459
    660000: 12.593
    690000: 13.098
    835000: 13.637
    860000: 13.973
"""