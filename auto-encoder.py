import torch
import torch.nn as nn
import os
import numpy as np

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

input_size = 6040
hidden_size = 500
output_size = 6040
num_epochs = 20
learning_rate = 0.001

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.encoder(x)
        out = self.tanh(out)
        out = self.decoder(out)
        return out

net = Net(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



import dataset

for epoch in range(num_epochs):
    for i, (id,sample) in enumerate(dataset.train_loader):

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(sample)
        loss = criterion(outputs, sample)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                   % (epoch + 1, num_epochs, i + 1, len(dataset.train_loader) // dataset.train_loader.batch_size, loss.data[0]))


