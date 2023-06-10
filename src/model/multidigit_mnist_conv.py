# This file is a copy with (minor modification) from https://github.com/Felix-Petersen/diffsort

import torch
from torch import nn
import torch.nn.functional as F


class MultiDigitMNISTConv(nn.Module):
    def __init__(self, n_digits=4):
        super(MultiDigitMNISTConv, self).__init__()
        self.n_digits = n_digits

        self.hidden_dim_1 = 32
        self.hidden_dim_2 = 32
        self.conv1 = nn.Conv2d(1, self.hidden_dim_1, 5, 1, 2)
        self.conv2 = nn.Conv2d(self.hidden_dim_1, self.hidden_dim_2, 5, 1, 2)
        self.fc1 = nn.Linear(n_digits * 7 * 7 * self.hidden_dim_2, self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, 1)

    def forward(self, x):
        x_shape = x.shape
        if len(x_shape) == 5:
            x = x.reshape(-1, *x_shape[2:])
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.n_digits * 7 * 7 * self.hidden_dim_2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(*x_shape[:2], 1)
        return x
