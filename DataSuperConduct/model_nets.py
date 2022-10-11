# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 23:34
# @Author  : SimonZHENG
# @Email   : zhengsiming2016@163.com
# @File    : model_nets.py
# @Software: PyCharm
# basic functions
import os
import sys
import math
import numpy as np
import random
import argparse
import shutil
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# torch functions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


class ConcatLayer(nn.Module):
    """Concate two tensors along the dim axis"""
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *input_xy):
        for module in self._modules.values():
            if isinstance(input_xy, tuple):
                input_xy = module(*input_xy)
            else:
                input_xy = module(input_xy)
        return input_xy


class MIDiscriminator(nn.Module):
    """Construct the mi-discriminator network with input dimension (u_dim + y_dim)"""
    def __init__(self, u_dim, y_dim, width_vec: list = None):
        """
        Args:
            u_dim (int): The dimension of the learned representation.
            y_dim (int): The response vector dimension.
            width_vec (list): The list containing the widths in all layers, except the output layer.
        """
        super(MIDiscriminator, self).__init__()
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.width_vec = width_vec

        modules = []
        if width_vec is None:
            width_vec = [u_dim + y_dim, 16, 8]

        # Network
        for i in range(len(width_vec) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(width_vec[i], width_vec[i + 1]),
                    nn.LeakyReLU(0.2)))

        T = nn.Sequential(*modules, nn.Linear(width_vec[-1], 1))

        self.net = CustomSequential(ConcatLayer(), *T)

    def forward(self, r, y):
        d_out = self.net(r, y)
        return d_out


class PushDiscriminator(nn.Module):
    """Construct the push-discriminator network"""
    def __init__(self, u_dim, width_vec: list = None):
        """
        Args:
            u_dim (int): The input dimension, which is also the dimension of the reference distribution.
            width_vec (list): The list containing the widths in all layers, except the output layer.
        """
        super(PushDiscriminator, self).__init__()
        self.u_dim = u_dim
        self.width_vec = width_vec

        modules = []
        if width_vec is None:
            width_vec = [u_dim, 16, 8]

        # Network
        for i in range(len(width_vec) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(width_vec[i], width_vec[i + 1]),
                    nn.LeakyReLU(0.2)))

        self.net = nn.Sequential(*modules, nn.Linear(width_vec[-1], 1))

    def forward(self, r):
        q_out = self.net(r)
        return q_out


class Representer(nn.Module):
    """Construct the representer network"""
    def __init__(self, x_dim, u_dim, width_vec: list = None):
        """
        Args:
            x_dim (int): The input dimension.
            u_dim (int): The output dimension, which is also the dimension of the reference distribution.
            width_vec (list): The list containing the widths in all layers, except the output layer.
        """
        super(Representer, self).__init__()
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.width_vec = width_vec

        modules = []
        if width_vec is None:
            width_vec = [x_dim, 32, 16, 8]

        # Network
        for i in range(len(width_vec) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(width_vec[i], width_vec[i + 1]),
                    nn.LeakyReLU(0.2)))

        self.net = nn.Sequential(*modules, nn.Linear(width_vec[-1], u_dim))

    def forward(self, x):
        r_out = self.net(x)
        return r_out


def pairwise_distances(X, Y=None):
    """
    Calculate the distance matrix between the points in data matrices X and Y or X and X if Y is none.
    Args:
        X (float): An n*p data matrix stored in tensor data type.
        Y (float): An n*p data matrix stored in tensor data type.
    """
    x_norm = (X ** 2).sum(1).view(-1, 1)
    if Y is not None:
        y_norm = (Y ** 2).sum(1).view(1, -1)
    else:
        Y = X
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    return dist


def cor(X, Y, n, device):
    """
    Calculate the empirical distance correlation between X and Y.
    Args:
        X (float): An n*p data matrix stored in tensor data type.
        Y (float): An n*q data matrix stored in tensor data type.
        n (int): Sample size.
        device: Computation device: CPU or GPU
    """
    DX = pairwise_distances(X)
    DY = pairwise_distances(Y)
    J = (torch.eye(n) - torch.ones(n, n) / n).to(device)
    RX = J @ DX @ J
    RY = J @ DY @ J
    covXY = torch.mul(RX, RY).sum() / (n * n)
    covX = torch.mul(RX, RX).sum() / (n * n)
    covY = torch.mul(RY, RY).sum() / (n * n)
    return covXY / torch.sqrt(covX * covY)


def torch_tile(tensor, dim, n):
    """Tile n times along the dim axis"""
    if dim == 0:
        return tensor.unsqueeze(0).transpose(0, 1).repeat(1, n, 1).view(-1, tensor.shape[1])
    else:
        return tensor.unsqueeze(0).transpose(0, 1).repeat(1, 1, n).view(tensor.shape[0], -1)


def weight_init(m):
    """
    Initialize the eight of the neural network m.
    Args:
        m (nn.Module): An nn.Module object.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def npLoader(Loader, net, device):
    """
    Obtain the features and corresponding targets after representation learning.
    Args:
        Loader (DataLoader): An DataLoader object.
        net (nn.Module): An nn.Module object in device.
        device: Computation device: CPU or GPU
    """
    y, X, _ = next(iter(Loader))
    mb_size = X.shape[0]
    X = net(X.to(device)).cpu().detach().numpy()
    y = y.numpy()
    torch.cuda.empty_cache()
    for step, (y_t, X_t, _) in enumerate(Loader):
        X_t = net(X_t.to(device)).cpu().detach().numpy()
        y_t = y_t.numpy()
        X = np.concatenate((X, X_t))
        y = np.concatenate((y, y_t))
        torch.cuda.empty_cache()
    return X[mb_size:], y[mb_size:]


def train(trainLoader, R_net, D_net, Q_net, optimizer_R, optimizer_D, optimizer_Q, device):
    """
    Training procedure for representation learning.
    Args:
        Loader (DataLoader): An DataLoader object.
        R_net, D_net, Q_net (nn.Module): Three nn.Module objects for representer, MI-Discriminator, and
                                         Push-Discriminator in device.
        optimizer_R, optimizer_D, optimizer_Q (torch.optim): The corresponding optimizers.
        device: Computation device: CPU or GPU
    """
    R_net.train()
    D_net.train()
    Q_net.train()
    for batch_idx, (y_data, x_data, u_data) in enumerate(trainLoader):
        x_data = Variable(x_data.to(device))
        y_data = Variable(y_data.to(device))
        u_data = Variable(u_data.to(device))

        # Discriminator forward-loss-backward-update
        laten_r = R_net(x_data)
        new_laten_r = Variable(laten_r.clone())

        # MI-Discriminator forward-loss-backward-update
        index = torch.randperm(x_data.shape[0])
        permuted_y_data = y_data[index, :]
        optimizer_D.zero_grad()
        D_loss = torch.mean(torch.exp(D_net(new_laten_r, permuted_y_data))) - torch.mean(D_net(new_laten_r, y_data))
        D_loss.backward()
        optimizer_D.step()

        # Push-Discriminator forward-loss-backward-update
        optimizer_Q.zero_grad()
        Q_loss = torch.mean(torch.exp(Q_net(u_data))) - torch.mean(Q_net(new_laten_r))
        Q_loss.backward()
        optimizer_Q.step()

        # Representer forward-loss-backward-update
        optimizer_R.zero_grad()

        R_loss = (torch.mean(torch.exp(D_net(R_net(x_data), permuted_y_data))) -
                  torch.mean(D_net(R_net(x_data), y_data))) - 2 * (torch.mean(torch.exp(Q_net(u_data)))
                                                                   - torch.mean(Q_net(R_net(x_data))))
        R_loss.backward()
        optimizer_R.step()


def test(R_net, testLoader, device):
    """
    Calculate the the empirical distance correlation between the features and corresponding targets
    after representation learning based on the DataLoader object testLoader.
    Args:
        R_net (nn.Module): The learned feature mapping, which is an nn.Module object in device.
        testLoader (DataLoader): An DataLoader object.
        device: Computation device: CPU or GPU
    """
    R_net.eval()
    dCor_loss = 0
    with torch.no_grad():
        for target, data, _ in testLoader:
            data = Variable(data.to(device))
            target = Variable(target.to(device))
            latent = R_net(data)
            dCor_loss += cor(latent, target, data.shape[0], device)
    dCor_loss /= len(testLoader)
    return dCor_loss.cpu().detach().numpy().item()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=15, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 15
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = None
        self.delta = delta
        self.epoch = 0

    def __call__(self, custom_path, val_loss, net_r, epoch):

        if self.path is None:
            self.path = custom_path
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(net_r, epoch)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(net_r, epoch)
            self.counter = 0

    def save_checkpoint(self, net_r, epoch):
        """Saves model when validation loss decrease."""
        self.epoch = epoch
        torch.save(net_r.state_dict(), os.path.join(self.path, 'R.pt'))


def setup_seed(seed, cuda):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): Random seed.
        cuda (bool): If TURE, set the random seed for computation steps involving GPU.
    """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
