import torch
import torch.nn as nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import mmread


DATA_DIR = './ml-1m-5cv'
batch_size = 100


# train = mmread(os.path.join(DATA_DIR,'train.%s'%cv)).A
# test = mmread(os.path.join(DATA_DIR,'test.%s'%cv)).A
# targets = np.unique(test.nonzero()[0])

class MLIRDataset(Dataset):
    def __init__(self, root_dir, cv, part, tranform = None):
        self.part = part

        if self.part == 'train':
            self.train = mmread(os.path.join(root_dir, 'train.%s' % cv)).A.astype(np.float32)
            self.newusers = mmread(os.path.join(root_dir, 'input.%s' % cv)).A.astype(np.float32)
            self.known = self.train + self.newusers
            self.train_users = np.unique(self.train.nonzero()[0])
            # self.train = self.train + self.newusers
        elif self.part == 'test':
            self.test = mmread(os.path.join(root_dir, 'eval.%s' % cv)).A.astype(np.float32)
            self.targets = np.unique(self.test.nonzero()[0])

        else:
            raise AttributeError('which part of data to fetch?')

        self.transform = tranform

    def __len__(self):
        return self.train.shape[1]

    def __getitem__(self, idx):
        if self.part == 'train':
            sample = self.known[:,idx]
        elif self.part == 'test':
            sample = self.test[:,idx]
        elif self.part == 'all':
            sample = self.data[:,idx]
        else:
            raise AttributeError('which part of data to fetch?')
        if self.transform:
            sample = self.transform(sample)
        return {'id':idx,'sample':sample}


class MyDataset():
    def __init__(self, train_dataset,test_dataset,train_loader, test_loader):
        self.train_dataset = train_dataset
        self.test_dataset  = test_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader

def load_data(cv):

    train_dataset = MLIRDataset(DATA_DIR,cv,'train')
    test_dataset = MLIRDataset(DATA_DIR,cv, 'test')
    train_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4,
                                        pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                        pin_memory = True)

    return MyDataset(train_dataset,test_dataset,train_loader, test_loader)
#
# for i_batch, sample_batched in enumerate(train_loader):
#     print i_batch,sample_batched