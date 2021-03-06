import argparse
import yaml
import utils
from torch.utils.data import DataLoader
import os
from os.path import join, exists
import shutil

if __name__ == '__main__':
    # set argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', default='baxter')
    parser.add_argument('--config', default=None, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # import config
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    mode = config['mode']
    if mode in ['pose_fk']:
        task_config = config['fk']
    cfg_stg = task_config['strategy']

    # create save dir and copy config
    if not exists(join(cfg_stg['save_path'], 'checkpoint')):
        os.makedirs(join(cfg_stg['save_path'], 'checkpoint'))
    shutil.copy(args.config, join(cfg_stg['save_path'], 'config.yaml'))
    print('Results are saved at ', cfg_stg['save_path'])

    # build dataloader
    cfg_data = task_config['dataset']
    transform = cfg_data.get('augmentation', None)
    if transform is not None:
        transform = utils.build_augmentation(cfg_data)
    trainset, testset = utils.build_dataset(cfg_data, transform)
    train_dataloader = DataLoader(trainset, batch_size=cfg_data['train_set']['batch_size'], shuffle=True, num_workers=4)
    test_dataloader = DataLoader(testset, batch_size=cfg_data['test_set']['batch_size'], shuffle=False, num_workers=4)

    # build trainer
    trainer = utils.build_trainer(mode, task_config, args.device, args.resume or args.test_only)

    # train and test
    if args.test_only:
        trainer.test(trainer.start_epoch-1, test_dataloader)
    else:
        for epoch in range(trainer.start_epoch, cfg_stg['total_epochs']+1):
            trainer.train(epoch, train_dataloader)
            if epoch % 10 == 0:
                trainer.test(epoch, test_dataloader)
                trainer.save_model(epoch)


