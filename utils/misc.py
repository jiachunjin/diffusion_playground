import os
import argparse
import torch
import yaml
import datetime

from model.score_net import EDM_Net

def is_master():
    return int(os.environ["LOCAL_RANK"]) == 0

def load_scorenet(ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    net_config = state['net_config']
    net = EDM_Net(**net_config)
    net.load_state_dict(state['net'])
    return net

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