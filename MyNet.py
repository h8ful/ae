# %matplotlib inline
import matplotlib.pyplot as plt

from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import mmread

import pandas as pd

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_size, output_size)
        self.repres = dict()

    def forward(self, x):
        out = self.encoder(x)
        self.repres['hidden'] = out.clone()
        out = self.relu(out)
        self.repres['hidden_relu'] = out.clone()
        out = self.decoder(out)
        self.repres['output'] = out.clone()
        return out

