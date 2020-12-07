import torch
from torch.utils.data import Dataset
import numpy as np
import os


class BaxterDataset(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = self.parse_data(data_path)

    def parse_data(self, data_path):
        assert os.path.exists(data_path), '{} not exists'.format(data_path)
        fdata = open(data_path, 'r')
        data = []
        for line in fdata.readlines():
            line = line.strip().split(' ')
            line = list(map(float, line))
            data.append(line)
        data = np.array(data)
        return data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        ang, pos, ori = data[0:7], data[7:11], data[11:14]
        return ang, pos, ori
