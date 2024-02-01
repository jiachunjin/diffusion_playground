import os
import shutil
import logging
import argparse

import torch
from tensorboard import program

from trainer import Score_Trainer
from utils.misc import create_exp_name, load_config, str2bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, required=True)
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

def main(arg):
    # Create the directory for the current experiment
    name = create_exp_name(arg.exp_name)
    exp_dir = os.path.join('experiments', name)
    log_dir = os.path.join('./', exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    sample_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(exp_dir)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(sample_dir)
    config_path = os.path.join('config', f'{arg.config}.yaml')
    config = load_config(config_path)
    shutil.copyfile(config_path, os.path.join(exp_dir, f'{arg.config}.yaml'))

    # Create the training logger
    level = getattr(logging, args.verbose.upper(), None)
    my_logger = get_logger(level)

    # Setup tensorboard
    if str2bool(args.tensorboard):
        tb = program.TensorBoard()
        tb.configure(argv=[None, f'--logdir={log_dir}', f'--port={args.tb_port}', f'--load_fast=false'])
        url = tb.launch()

    # Fill in more training information
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    my_logger.info(f'Using device: {device}')
    config['device'] = device
    config['data']['name'] = arg.data # mainly used for toy_data
    
    # Train
    trainer = Score_Trainer(config, my_logger, log_dir, ckpt_dir, sample_dir)
    my_logger.info('Start to train')
    trainer.train()
    my_logger.info('Training Done')


if __name__ == '__main__':
    args = parse_args()
    main(args)

# python main.py -n test_unet -d cifar10 -c cifar10 --verbose info --tensorboard false