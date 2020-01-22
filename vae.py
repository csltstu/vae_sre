#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: vae.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Wed 22 Jan 2020 09:25:10 AM CST
# ************************************************************************/



import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, z_dim=50):
        super().__init__()
        self.z_dim = z_dim

        # for encoder
        self.fc1 = nn.Linear(72, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)

        self.fc21 = nn.Linear(500, self.z_dim)  # fc21 for mean of Z
        self.fc22 = nn.Linear(500, self.z_dim)  # fc22 for log variance of Z

        # for decoder
        self.fc4 = nn.Linear(self.z_dim, 500)
        self.fc5 = nn.Linear(500, 500)
        self.fc6 = nn.Linear(500, 500)
        self.fc7 = nn.Linear(500, 72)

    def encode(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std

    def decode(self, z):
        y = F.relu(self.fc4(z))
        y = F.relu(self.fc5(y))
        y = F.relu(self.fc6(y))
        return self.fc7(y)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

