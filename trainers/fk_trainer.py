import torch
import utils
import os
from os.path import join, exists
import torch.nn.functional as F
from .base_trainer import BaseTrainer


class PoseFKTrainer(BaseTrainer):

    def __init__(self, config, device, resume=False):
        super().__init__(config, device, resume)

    def train(self, epoch, train_dataloader):
        self.model.train()
        lr = utils.update_lr(epoch, self.cfg_stg, self.optimizer)
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
            self.optimizer.step()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            batch_num += 1

        self.tb_logger.add_scalar('train_loss_pos', total_loss1 / batch_num, epoch)
        self.tb_logger.add_scalar('train_loss_ori', total_loss2 / batch_num, epoch)
        if epoch % 1 == 0:
            self.logger.info('Train: epoch {}, lr {:.6f}, loss_pos {:.6f}, loss_ori {:.6f}'.format(
                epoch, lr, total_loss1/batch_num, total_loss2/batch_num))

    def test(self, epoch, test_dataloader):
        self.model.eval()
        total_loss1 = 0
        total_loss2 = 0
        batch_num = 0
        for ang, pos, ori in test_dataloader:
            ang, pos, ori = ang.to(self.device), pos.to(self.device), ori.to(self.device)

            pred_pos, pred_ori_pow2 = self.model(ang)

            loss1, loss2 = self.criterion(pos, ori, pred_pos, pred_ori_pow2)
            loss = loss1 + loss2
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            batch_num += 1
        self.tb_logger.add_scalar('test_loss_pos', total_loss1 / batch_num, epoch)
        self.tb_logger.add_scalar('test_loss_ori', total_loss2 / batch_num, epoch)
        self.logger.info('Test: epoch {}, loss_pos {:.6f}, loss_ori {:.6f}'.format(
            epoch, total_loss1/batch_num, total_loss2/batch_num))

    def criterion(self, pos, ori, pred_pos, pred_ori_pow2):
        loss1 = F.mse_loss(pred_pos, pos)
        loss2 = self.kl_div(ori**2, pred_ori_pow2)
        return loss1, loss2

    def kl_div(self, p_input, p_target):
        kl_div = p_target * (torch.log(p_target + 1e-8) - torch.log(p_input + 1e-8))
        kl_div = kl_div.sum(dim=1).mean(dim=0)
        return kl_div
