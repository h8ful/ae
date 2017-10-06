import torch
import torch.nn as nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import mmread


# DATA_DIR = './attributes'
batch_size = 100


# train = mmread(os.path.join(DATA_DIR,'train.%s'%cv)).A
# test = mmread(os.path.join(DATA_DIR,'test.%s'%cv)).A
# targets = np.unique(test.nonzero()[0])

class ATTRDataset(Dataset):
    def __init__(self,path ):
        self.attributes = mmread(path).A
        self.path = path

    def __len__(self):
        return self.train.shape[1]

    def __getitem__(self, idx):
        return self.attributes[idx]

class MyDataset():
    def __init__(self, attr_dataset,attr_loader):
        self.attr_dataset = attr_dataset
        self.attr_loader  = attr_loader

def load_attributes(path):

    attr_dataset = ATTRDataset(path)
    attr_loader = DataLoader(dataset=attr_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=4)
    return MyDataset(attr_dataset,attr_loader)
#
# for i_batch, sample_batched in enumerate(train_loader):
#     print i_batch,sample_batched