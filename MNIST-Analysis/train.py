#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: SimonZHENG
# datetime:2022/9/2111:55
# software: PyCharm
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import argparse

from model_nets import *

# arguments setting
parser = argparse.ArgumentParser(description='PyTorch Implementation of Representation Learning for MNIST')
parser.add_argument('--train', default=3000, type=int, help='size of training dataset')
parser.add_argument('--val', default=1000, type=int, help='size of validation dataset')
parser.add_argument('--test', default=1000, type=int, help='size of testing dataset')
parser.add_argument('--batchSz', type=int, default=100, help='mini batch size')
parser.add_argument('--latent_dim', type=int, default=16, help='the dimension of latent space')
parser.add_argument('--nEpochs', type=int, default=300, help='the number of outer loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=42, help='random seed') # 123
parser.add_argument('--lr', type=float, default=3e-3)
args = parser.parse_args([])
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.latent_dim, args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'Results/MNIST'
setup_seed(args.seed, args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

net_saved_path = args.save+'/SavedNet'
if not os.path.exists(net_saved_path):
    os.makedirs(net_saved_path, exist_ok=True)

data_saved_path = args.save+'/SavedData'
if not os.path.exists(data_saved_path):
    os.makedirs(data_saved_path, exist_ok=True)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

mnist_train = dset.MNIST(root='mnist', train=True, download=True, transform=T.ToTensor())
mnist_test = dset.MNIST(root='mnist', train=False, download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=len(mnist_train))
loader_test = DataLoader(mnist_test, batch_size=len(mnist_test))

train_data = enumerate(loader_train)
_, (train_x, train_label) = next(train_data)

test_data = enumerate(loader_test)
_, (test_x, test_label) = next(test_data)

train_new_label = (train_label >= 5) + 0.0
test_new_label = (test_label >= 5) + 0.0

mnist_train = TensorDataset(train_x, train_new_label)
mnist_test = TensorDataset(test_x, test_new_label)

tup1 = ()
tup2 = ()
rep_num = 5  # The number of the repeated subsamplings.
ini_num = 5  # The number of the repeated weight initializations.
setup_seed(42, args.cuda)
# Save the subsamples for GSIR and GSAVE.
for k in range(rep_num):
    loader_train = DataLoader(mnist_train, batch_size=args.train, sampler=ChunkSampler(args.train))

    loader_val = DataLoader(mnist_train, batch_size=args.val, sampler=ChunkSampler(args.val))

    loader_test = DataLoader(mnist_test, batch_size=args.test, sampler=ChunkSampler(args.test, 0, 10000))

    train_L = enumerate(loader_train)
    _, (train_data, train_label) = next(train_L)

    validation = enumerate(loader_val)
    _, (val_data, val_label) = next(validation)

    test = enumerate(loader_test)
    _, (test_data, test_label) = next(test)

    torch.save(train_data, data_saved_path + '/train_data(' + str(k) + ').pty')
    torch.save(train_label, data_saved_path + '/train_label(' + str(k) + ').pty')
    torch.save(val_data, data_saved_path + '/val_data(' + str(k) + ').pty')
    torch.save(val_label, data_saved_path + '/val_label(' + str(k) + ').pty')
    torch.save(test_data, data_saved_path + '/test_data(' + str(k) + ').pty')
    torch.save(test_label, data_saved_path + '/test_label(' + str(k) + ').pty')

u_dim = args.latent_dim

for k in range(rep_num):
    train_data = torch.load(data_saved_path + '/train_data(' + str(k) + ').pty')
    train_label = torch.load(data_saved_path + '/train_label(' + str(k) + ').pty')
    val_data = torch.load(data_saved_path + '/val_data(' + str(k) + ').pty')
    val_label = torch.load(data_saved_path + '/val_label(' + str(k) + ').pty')
    test_data = torch.load(data_saved_path + '/test_data(' + str(k) + ').pty')
    test_label = torch.load(data_saved_path + '/test_label(' + str(k) + ').pty')

    U_train = torch.rand(args.train, args.latent_dim)
    U_validation = torch.rand(args.val, args.latent_dim)
    U_test = torch.rand(args.test, args.latent_dim)

    train_dat = TensorDataset(train_label, train_data, U_train)
    trainLoader = DataLoader(train_dat, batch_size=args.batchSz, shuffle=True)

    validation_dat = TensorDataset(val_label, val_data, U_validation)
    validationLoader = DataLoader(validation_dat, batch_size=len(validation_dat), shuffle=False)

    test_dat = TensorDataset(test_label, test_data, U_test)
    testLoader = DataLoader(test_dat, batch_size=args.batchSz, shuffle=False)

    best_dc = None
    best_R_net = DenseNet(lant_dim=u_dim, growthRate=12, depth=20, reduction=0.5, bottleneck=True)
    if args.cuda:
        best_R_net = best_R_net.cuda()
    X_test_t, y_test_t = test_data.to(device), test_label.to(device)
    for i in range(ini_num):
        R_net = DenseNet(lant_dim=u_dim, growthRate=12, depth=20, reduction=0.5, bottleneck=True)
        D_width_vec = [u_dim, 32, 32]
        D_net1 = Discriminator(u_dim, D_width_vec)
        D_net2 = Discriminator(u_dim, D_width_vec)
        Q_width_vec = [u_dim, 32, 32]
        Q_net = Discriminator(u_dim, Q_width_vec)

        D_net1.apply(weight_init)
        D_net2.apply(weight_init)
        Q_net.apply(weight_init)

        if args.cuda:
            R_net = R_net.cuda()
            D_net1 = D_net1.cuda()
            D_net2 = D_net2.cuda()
            Q_net = Q_net.cuda()

        # default weight decay parameter
        wd = 1e-5
        # user-selected learning rate
        mylr = args.lr

        optimizer_R = optim.Adam(R_net.parameters(), lr=mylr, weight_decay=wd)
        optimizer_D1 = optim.Adam(D_net1.parameters(), lr=mylr, weight_decay=wd)
        optimizer_D2 = optim.Adam(D_net2.parameters(), lr=mylr, weight_decay=wd)
        optimizer_Q = optim.Adam(Q_net.parameters(), lr=3e-4, weight_decay=wd)

        patience = 50
        early_stopping = EarlyStopping(patience, verbose=True)
        for epoch in range(1, args.nEpochs + 1):
            train(trainLoader, R_net, D_net1, D_net2, Q_net, optimizer_R, optimizer_D1, optimizer_D2, optimizer_Q, device)
            dc_loss = test_fun(R_net, validationLoader, device)
            early_stopping(args.save, -1 * dc_loss, R_net, epoch)
            if early_stopping.early_stop:
                print("Early stopping at Epoch:", early_stopping.epoch)  # stop training
                break
        print('Init', i, '. When Stopping, DC is', -1 * early_stopping.best_score, 'and Iteration is',
              early_stopping.epoch)
        # Save the best representer network
        if best_dc is None or -1 * early_stopping.best_score > best_dc:
            model_path = os.path.join(args.save, 'R.pt')
            R_net.load_state_dict(torch.load(model_path))
            best_dc = -1 * early_stopping.best_score
            torch.save(R_net.state_dict(), os.path.join(net_saved_path, 'best_R'+ str(k) +'.pt'))
        del R_net
        del D_net1
        del D_net2
        del Q_net
        torch.cuda.empty_cache()
    best_model_path = os.path.join(net_saved_path, 'best_R'+ str(k) +'.pt')
    best_R_net.load_state_dict(torch.load(best_model_path))
    best_R_net.eval()
    X_train_tf, y_train_tf = npLoader(trainLoader, best_R_net, device)
    X_test_tf, y_test_tf = npLoader(testLoader, best_R_net, device)
    y_train_tf = y_train_tf.ravel()
    y_test_tf = y_test_tf.ravel()
    scaler = StandardScaler()
    scaler.fit(X_train_tf)
    X_train_tf = scaler.transform(X_train_tf)
    X_test_tf = scaler.transform(X_test_tf)

    # Validation via KNN classifier with k=5
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_tf, y_train_tf)
    y_pred = classifier.predict(X_test_tf)
    acc = 100 * np.sum(y_pred == y_test_tf) / y_pred.shape
    tup1 += (acc,)

    # The empirical DC calculation
    lant_t = best_R_net(X_test_t)
    tup2 += (cor(lant_t, y_test_t, X_test_t.shape[0], device).item(),)
    torch.cuda.empty_cache()
print("done!")
print('latent_dim-', args.latent_dim)
print('MSRL ACC: {:.2f}({:.2f})\n'.format(np.mean(tup1), np.var(tup1) ** 0.5))
print('MSRL DC: {:.2f}({:.2f})\n'.format(np.mean(tup2), np.var(tup2) ** 0.5))
print(tup1, tup2)
