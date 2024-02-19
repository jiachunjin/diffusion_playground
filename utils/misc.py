import os
import argparse
import torch
import yaml
import datetime

from model.score_net import EDM_Net

def is_master():
    return int(os.environ["LOCAL_RANK"]) == 0

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def load_scorenet(ckpt_path, use_ema=False):
    state = torch.load(ckpt_path, map_location='cpu')
    net_config = state['net_config']
    net = EDM_Net(**net_config)
    if use_ema:
        net.load_state_dict(state['ema'])
    else:
        net.load_state_dict(state['net'])
    return net, state['history_iters']

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dataloader_one_batch(dataset):
    x_list = []
    y_list = []
    for b, (x, y) in enumerate(dataset):
        x_list.append(x)
        y_list.append(y)
    return torch.cat(x_list), torch.cat(y_list)

def create_exp_name(comments=''):
    x = datetime.datetime.now()
    name = x.strftime("%y%m%d") + '-' + x.strftime("%X")
    if len(comments) > 0:
        name = comments + '_' + name
    return name

def load_config(path):
    with open(path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')