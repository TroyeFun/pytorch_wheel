import torch
import logging
import os.path
import torch.optim as optim
from models import BaxterFK
from datasets import BaxterDataset
from trainers import PoseFKTrainer


def build_model(cfg_model):
    model_type = cfg_model['type']
    model_classes = {'baxter_fk': BaxterFK}
    kwargs = cfg_model['kwargs']
    model = model_classes[model_type](**kwargs)
    return model


def build_optimizer(cfg_stg, model, start_epoch):
    lr = update_lr(start_epoch, cfg_stg)
    momentum = cfg_stg['momentum']
    weight_decay = cfg_stg['weight_decay']
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    return optimizer


def build_augmentation(cfg_data):
    # TODO
    pass


def build_dataset(cfg_data, transform):
    if cfg_data['type'] == 'baxter':
        dataset_class = BaxterDataset
    trainset = dataset_class(cfg_data['train_set']['path'], transform)
    testset = dataset_class(cfg_data['test_set']['path'], transform)
    return trainset, testset


def build_trainer(mode, task_config, device, resume):
    if mode == 'pose_fk':
        trainer_class = PoseFKTrainer
    trainer = trainer_class(task_config, device, resume)
    return trainer


def create_logger(save_path, level=logging.INFO):
    logger = logging.getLogger('global_logger')
    formatter = logging.Formatter('[%(asctime)s][%(filename)15s]'
                                  '[line:%(lineno)4d][%(levelname)8s]%(message)s')
    log_path = os.path.join(save_path, 'log.txt')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def update_lr(cur_epoch, cfg_stg, optimizer=None):
    lr_steps = cfg_stg['lr_steps']
    lr_mults = cfg_stg['lr_mults']
    lr = cfg_stg['base_lr']

    idx = 0
    while idx < len(lr_steps) and cur_epoch > lr_steps[idx]:
        lr *= lr_mults[idx]
        idx += 1
    if idx == len(lr_steps):
        return lr
    if cur_epoch == lr_steps[idx]:
        lr *= lr_mults[idx]
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return lr



