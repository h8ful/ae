from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import  manifold

import pandas as pd

import rec3

import matplotlib.pyplot as plt
from pprint import pprint
import copy

def weighted_average(input, targets, weights):
    return nn.functional.mse_loss(input.mul(weights), targets.mul(weights),size_average=False)*1.0/weights.sum()
def rec_pred(pred_ratings, train, test, targets):
    pred = rec3.Rec()
    pred.set_prediction_matrix(train, pred_ratings)
    pred.produce_rec_list(train,targets=targets)
    pred.evaluate(test = test, rec_len=5)
    return pred


class NetNormWeights(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NetNormWeights, self).__init__()
        self.representations = dict()
        self.encoder = nn.utils.weight_norm(nn.Linear(input_size, hidden_size))
        self.decoder = nn.utils.weight_norm(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        out = self.encoder(x)
        self.representations['hidden'] = out.data.cpu().numpy().copy()
#         out = self.relu(out)
#         self.representations['hidden_relu'] = out.data.numpy().copy()
        out = self.decoder(out)
#         self.representations['decode'] = out.data.numpy().copy()
        return out

cv = 1
