import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import prepare_dataloader
from model.score_net import EDM_Net


class Base_Trainer():
    def __init__(self, config, log_dir, ckpt_dir, sample_dir):
        self.config = config
        self.device = config['device']
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.sample_dir = sample_dir
        self.num_iters = 0
        self.start_iter = 0
        self._build_dataloader()

    def _build_dataloader(self):
        root = self.config['data']['root']
        name = self.config['data']['name']
        train_batch = test_batch = self.config['data']['batch_size']
        num_workers = self.config['data']['num_workers']

        logging.info('Start to build dataloader')
        self.train_loader, self.test_loader = prepare_dataloader(root, name, train_batch, test_batch, num_workers)
        logging.info('Dataloader built')

    def _build_scorenet(self):
        self.net = EDM_Net(**self.config['score_net']).to(self.device)
        logging.info('Network built')

    def _build_optimizer(self):
        config = self.config['optimizer']
        if config['type'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=config['lr'],
                                              weight_decay=config['weight_decay'],
                                              betas=(config['beta1'], 0.999),
                                              amsgrad=config['amsgrad'],
                                              eps=config['eps'])
        else:
            raise NotImplementedError
