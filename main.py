import os
import shutil
import argparse

from trainer import DDPM_Trainer
from utils import create_exp_name, load_config, str2bool


def run(arg):
    name = create_exp_name(arg.exp_name)
    exp_dir = os.path.join('experiments', name)
    log_dir = os.path.join(exp_dir, 'logs')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    sample_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(exp_dir)
    os.makedirs(log_dir)
    os.makedirs(ckpt_dir)
    os.makedirs(sample_dir)
    print(f'\nlog_dir: {log_dir}\n')

    config_path = os.path.join('config', 'ddpm.yaml')
    config = load_config(config_path)
    shutil.copyfile(config_path, os.path.join(exp_dir, 'config.yaml'))

    if args.data == 'mnist':
        config['data_config']['name'] = 'mnist'
        config['data_config']['num_channel'] = 1
        config['data_config']['img_size'] = 32

    config['parametric'] = args.parametric
    config['K'] = args.K

    trainer = DDPM_Trainer(config, log_dir, ckpt_dir, sample_dir)

    print('Start to train')
    trainer.train()
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--exp_name', type=str, required=True)
    parser.add_argument('-d', '--data', type=str, choices=['mnist'], required=True)
    parser.add_argument('-k', '--K', type=int, default=1)
    parser.add_argument('-p', '--parametric', type=str2bool, help='use parametric mode or not')
    args = parser.parse_args()
    run(args)