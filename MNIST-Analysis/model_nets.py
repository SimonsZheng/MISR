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


class Bottleneck(nn.Module):
    """Bottleneck block for DenseNet structure"""
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    """BatchNorm-Conv2d block for DenseNet structure"""
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    """Transition block for DenseNet structure"""
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    """Construct the DenseNet neural network"""
    def __init__(self, lant_dim, growthRate, depth, reduction, bottleneck):
        super(DenseNet, self).__init__()
        self.lant_dim = lant_dim

        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks * growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc1 = nn.Linear(nChannels, self.lant_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 7))
        out = self.fc1(out)
        return out


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


def train(trainLoader, R_net, D_net1, D_net2, Q_net, optimizer_R, optimizer_D1, optimizer_D2, optimizer_Q, device):
    """
    Training procedure for representation learning.
    Args:
        trainLoader (DataLoader): An DataLoader object.
        R_net, D_net1, D_net2, Q_net (nn.Module): Three nn.Module objects for representer, two conditional
                                                  MI-Discriminator, and Push-Discriminator in device.
        optimizer_R, optimizer_D1, optimizer_D2, optimizer_Q (torch.optim): The corresponding optimizers.
        device: Computation device: CPU or GPU
    """
    R_net.train()
    D_net1.train()
    D_net2.train()
    Q_net.train()
    for batch_idx, (y_data, x_data, u_data) in enumerate(trainLoader):
        x_data = Variable(x_data.to(device))
        y_data = Variable(y_data.to(device))
        u_data = Variable(u_data.to(device))
        batch_size = y_data.shape[0]

        # Discriminator forward-loss-backward-update
        laten_r = R_net(x_data)
        new_laten_r = Variable(laten_r.clone())

        optimizer_D1.zero_grad()
        d1_temp = D_net1(new_laten_r)
        D1_loss = torch.mean(torch.exp(d1_temp)) - torch.sum(y_data * d1_temp) / torch.sum(y_data)
        D1_loss.backward()
        optimizer_D1.step()

        optimizer_D2.zero_grad()
        d2_temp = D_net2(new_laten_r)
        D2_loss = torch.mean(torch.exp(d2_temp)) - torch.sum((1 - y_data) * d2_temp) / torch.sum(1 - y_data)
        D2_loss.backward()
        optimizer_D2.step()

        # Push-Discriminator forward-loss-backward-update
        optimizer_Q.zero_grad()
        Q_loss = torch.mean(torch.exp(Q_net(u_data))) - torch.mean(Q_net(new_laten_r))
        Q_loss.backward()
        optimizer_Q.step()

        # Representer forward-loss-backward-update
        optimizer_R.zero_grad()

        laten_r = R_net(x_data)
        d1_temp = D_net1(laten_r)
        d2_temp = D_net2(laten_r)
        R_loss = (torch.mean(torch.exp(d1_temp)) - torch.sum(y_data * d1_temp) / torch.sum(y_data)) * (torch.sum(y_data) / batch_size) \
                 + (torch.mean(torch.exp(d2_temp)) - torch.sum((1 - y_data) * d2_temp) / torch.sum(1 - y_data)) * (torch.sum(1 - y_data) / batch_size) \
                 - 2 * (torch.mean(torch.exp(Q_net(u_data))) - torch.mean(Q_net(laten_r)))
        R_loss.backward()
        optimizer_R.step()

        del x_data
        del y_data
        del u_data
        del laten_r
        del new_laten_r
        del d1_temp
        del d2_temp
        gc.collect()
        torch.cuda.empty_cache()


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


class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: The number of desired data points
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0, end=60000):
        self.num_samples = num_samples
        self.start = start
        self.end = end

    def __iter__(self):
        ind = np.arange(self.end - self.start)
        np.random.shuffle(ind)
        return iter(ind[0:self.num_samples])

    def __len__(self):
        return self.num_samples


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
