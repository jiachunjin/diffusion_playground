import os
import shutil
import logging
import argparse

import torch
from tensorboard import program
from torch.distributed import init_process_group, destroy_process_group

from trainer import Score_Trainer, CD_Trainer
from utils.misc import create_exp_name, load_config, str2bool, is_master


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'cd'])
    parser.add_argument('-d', '--data', type=str, choices=['cifar10', '8gaussians', '2spirals', 'checkerboard'], required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--tensorboard', type=str, default='True')
    parser.add_argument('--tb_port', type=int, default=9990)
    parser.add_argument('--verbose', type=str, default='info')
    
    args = parser.parse_args()
    return args

def get_logger(level):
    handler1 = logging.StreamHandler()
    # handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    # handler2.setFormatter(formatter)
    my_logger = logging.getLogger('training_logger')
    my_logger.addHandler(handler1)
    # logger.addHandler(handler2)
    my_logger.setLevel(level)
    return my_logger

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def create_experiment(args):
    config_path = os.path.join('config', f'{args.config}.yaml')
    config = load_config(config_path)
    log_dir = ckpt_dir = sample_dir = None
    if is_master():
        name = create_exp_name(args.exp_name)
        exp_dir = os.path.join('experiments', name)
        log_dir = os.path.join('./', exp_dir, 'logs')
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        sample_dir = os.path.join(exp_dir, 'samples')
        os.makedirs(exp_dir)
        os.makedirs(log_dir)
        os.makedirs(ckpt_dir)
        os.makedirs(sample_dir)
        shutil.copyfile(config_path, os.path.join(exp_dir, f'{args.config}.yaml'))
    return config, log_dir, ckpt_dir, sample_dir

def setup_tensorboard(log_dir):
        tb = program.TensorBoard()
        tb.configure(argv=[None, f'--logdir={log_dir}', f'--port={args.tb_port}', f'--load_fast=false'])
        url = tb.launch()

def main(args):
    # setup distributed training
    ddp_setup()
    # Create the directory for the current experiment
    config, log_dir, ckpt_dir, sample_dir = create_experiment(args)
    # Create the training logger
    my_logger = get_logger(level = getattr(logging, args.verbose.upper(), None))
    # Setup tensorboard
    if str2bool(args.tensorboard) and is_master():
        setup_tensorboard(log_dir)
    # Fill in more training information
    # ================================================================================
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # my_logger.info(f'Using device: {device}')
    # config['device'] = device
    # ================================================================================
    config['data']['name'] = args.data # mainly used for toy_data
    # Train
    if args.mode == 'train':
        trainer = Score_Trainer(config, my_logger, log_dir, ckpt_dir, sample_dir)
    elif args.mode == 'cd':
        trainer = CD_Trainer(config, my_logger, log_dir, ckpt_dir, sample_dir)
    if is_master():
        my_logger.info('Start to train')
    trainer.train()
    if is_master():
        my_logger.info('Training Done')
    destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    main(args)

# python train.py -n test -d 8gaussians -c toy_2d --verbose info --tensorboard false
# CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py -n test -d cifar10 -c cifar10 --verbose info
    
# nohup sh multi_gpu_train.sh > ./log.txt 2>&1 &
"""
train:

CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=16 torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py -m train -n cbd -d checkerboard -c toy_2d --verbose info

consistency distillation:
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=16 torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py -m cd -n cd_checker -d checkerboard -c cd_2d --verbose info
"""