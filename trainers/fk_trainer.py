import torch
import utils
import os
from os.path import join, exists
import torch.nn.functional as F


class PoseFKTrainer:

    def __init__(self, config, device, resume=False):
        self.config = config
        self.cfg_stg = config['strategy']
        self.device = device

        self.model = utils.build_model(config['model'])
        self.model.to(device)

        self.logger = utils.create_logger(self.cfg_stg['save_path'])

        self.start_epoch = 0
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

    def train(self, epoch, train_dataloader):
        self.model.train()
        lr = update_lr(epoch, self.cfg_stg, self.optimizer)
        total_loss1 = 0
        total_loss2 = 0
        batch_num = 0
        for ang, pos, ori in train_dataloader:
            ang, pos, ori = ang.to(self.device), pos.to(self.device), ori.to(self.device)
            pred_pos, pred_ori_pow2 = self.model(ang)

            loss1, loss2 = self.criterion(pos, ori, pred_pos, pred_ori_pow2)
            loss = loss1 + loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.update()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            batch_num += 1

        if epoch % 1 == 0:
            self.logger.info('Train: epoch {}, lr {}, loss_pos {}, loss_ori {}'.format(
                epoch, lr, total_loss1/batch_num, total_loss2/batch_num))

        if epoch % 10 == 0:
            self.save_model(epoch)

    def test(self, epoch, test_dataloader):
        self.model.eval()
        total_loss1 = 0
        total_loss2 = 0
        batch_num = 0
        for ang, pos, ori in test_dataloader:
            pred_pos, pred_ori_pow2 = self.model(ang)

            loss1, loss2 = self.criterion(pos, ori, pred_pos, pred_ori_pow2)
            loss = loss1 + loss2
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            batch_num += 1
        self.logger.info('Test: epoch {}, loss_pos {}, loss_ori {}'.format(
            epoch, total_loss1/batch_num, total_loss2/batch_num))

    def criterion(self, pos, ori, pred_pos, pred_ori_pow2):
        loss1 = F.mse_loss(pred_pos, pos)
        loss2 = self.kl_div(ori**2, pred_ori_pow2)
        return loss1, loss2

    def kl_div(self, p_input, p_target):
        kl_div = p_target * (torch.log(p_target + 1e-8) - torch.log(p_input + 1e-8))
        kl_div = kl_div.sum(dim=1).mean(dim=0)
        return kl_div
