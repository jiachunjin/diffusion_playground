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
    parser.add_argument('-d', '--data', type=str, choices=['mnist', '8gaussians', '2spirals', 'checkerboard'], required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--tensorboard', type=str, default='True')
    parser.add_argument('--tb_port', type=int, default=9990)
    parser.add_argument('--verbose', type=str, default='info')
    
    args = parser.parse_args()
    return args

def main(arg):
    name = create_exp_name(arg.exp_name)
    exp_dir = os.path.join('experiments', name)
    log_dir = os.path.join('./', exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    sample_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(exp_dir)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(sample_dir)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    # handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    # handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    # logger.addHandler(handler2)
    logger.setLevel(level)
    logging.info(f'log_dir: {log_dir}')

    if str2bool(args.tensorboard):
        tb = program.TensorBoard()
        tb.configure(argv=[None, f'--logdir={log_dir}', f'--port={args.tb_port}'])
        url = tb.launch()

    config_path = os.path.join('config', f'{arg.config}.yaml')
    config = load_config(config_path)
    shutil.copyfile(config_path, os.path.join(exp_dir, f'{arg.config}.yaml'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f'Using device: {device}')
    config['device'] = device
    config['diffusion']['device'] = device
    config['data']['name'] = arg.data
    


    trainer = Score_Trainer(config, log_dir, ckpt_dir, sample_dir)

    logging.info('Start to train')
    trainer.train()
    logging.info('Training Done')


if __name__ == '__main__':
    args = parse_args()
    main(args)

# python main.py -n tem -d 2spirals -c toy_2d --verbose warning