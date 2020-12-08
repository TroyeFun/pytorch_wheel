import torch
import utils
import os
from os.path import join, exists
from abc import abstractmethod
from tensorboardX import SummaryWriter


class BaseTrainer:

    def __init__(self, config, device, resume=False):
        self.config = config
        self.cfg_stg = config['strategy']
        self.device = device

        self.model = utils.build_model(config['model'])
        self.model.to(device)

        self.logger = utils.create_logger(self.cfg_stg['save_path'])
        self.tb_logger = SummaryWriter(join(self.cfg_stg['save_path'], 'events'))

        self.start_epoch = 1
        if resume:
            self.load_model()
        self.optimizer = utils.build_optimizer(config['strategy'], self.model, self.start_epoch)

    def load_model(self):
        ckpt_path = join(self.cfg_stg['save_path'], 'checkpoint', self.cfg_stg['resume_model'])
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.start_epoch = ckpt['epoch'] + 1
        self.model.load_state_dict(ckpt['state_dict'])

    def save_model(self, epoch):
        ckpt_path = join(self.cfg_stg['save_path'], 'checkpoint', 'epoch_{}_pth.tar'.format(epoch))
        torch.save({'epoch': epoch, 'state_dict': self.model.state_dict()}, ckpt_path)

    @abstractmethod
    def train(self, epoch, train_dataloader):
        pass

    @abstractmethod
    def test(self, epoch, test_dataloader):
        pass