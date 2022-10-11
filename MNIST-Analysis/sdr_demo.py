# -*- coding: utf-8 -*-
# @Time    : 2022/3/27 3:07
# @Author  : SimonZHENG
# @Email   : zhengsiming2016@163.com
# @File    : sdr_demo.py
# @Software: PyCharm
from sdr_family import *
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import argparse
import os

# arguments setting
parser = argparse.ArgumentParser(description='PyTorch Implementation of Representation Learning for MNIST')
parser.add_argument('--train', default=3000, type=int, help='size of training dataset')
parser.add_argument('--val', default=1000, type=int, help='size of validation dataset')
parser.add_argument('--test', default=1000, type=int, help='size of testing dataset')
parser.add_argument('--batchSz', type=int, default=100, help='mini batch size')
parser.add_argument('--latent_dim', type=int, default=16, help='the dimension of latent space')
parser.add_argument('--nEpochs', type=int, default=1000, help='the number of outer loop')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=42, help='random seed') # 123
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args([])
print(args)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args.latent_dim, args.cuda)
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'Results/MNIST'

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

data_saved_path = args.save+'/SavedData'
if not os.path.exists(data_saved_path):
    os.makedirs(data_saved_path, exist_ok=True)

data_savedSDR_path = args.save+'/SavedSDRData'
if not os.path.exists(data_savedSDR_path):
    os.makedirs(data_savedSDR_path, exist_ok=True)

# The parameter for cross-validation to select the best hyper-parameters for the generalized SDR methods.
folds_num = 4
ep_seq = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
rho_seq = np.array([1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5])
ytype = 'categorical'
atype = 'identity'

gsir_tup1 = ()
gsir_tup2 = ()
gsave_tup1 = ()
gsave_tup2 = ()
rep_num = 5  # The number of repeated subsamplings.

for k in range(rep_num):
    print(k)
    train_data = torch.load(data_saved_path + '/train_data(' + str(k) + ').pty')
    train_label = torch.load(data_saved_path + '/train_label(' + str(k) + ').pty')
    val_data = torch.load(data_saved_path + '/val_data(' + str(k) + ').pty')
    val_label = torch.load(data_saved_path + '/val_label(' + str(k) + ').pty')
    test_data = torch.load(data_saved_path + '/test_data(' + str(k) + ').pty')
    test_label = torch.load(data_saved_path + '/test_label(' + str(k) + ').pty')

    train_data = train_data.reshape((args.train, -1))
    val_data = val_data.reshape((args.val, -1))
    test_data = test_data.reshape((args.test, -1))

    X_train = train_data.numpy()
    X_validation = val_data.numpy()
    X_test = test_data.numpy()
    y_train = train_label.numpy()
    y_validation = val_label.numpy()
    y_test = test_label.numpy()

    pre_scaler = StandardScaler()
    pre_scaler.fit(X_train)
    X_train = pre_scaler.transform(X_train)
    X_validation = pre_scaler.transform(X_validation)
    X_test = pre_scaler.transform(X_test)

    # Cross-validation to select the best hyper-parameters for the generalized SDR methods.
    ex, rho_x = cv_gsdr_para_select(X_validation, y_validation, folds_num, ep_seq, rho_seq)
    ey, rho_y = 0, 0

    # The generalized sliced inverse regression method.
    pred_test = gsir(X_train, X_test, y_train, ytype, atype, ex, ey, rho_x, rho_y, args.latent_dim)
    dc_gsir = test_gsdr(pred_test, y_test, device)

    train_val_pred = gsir(X_train, X_train, y_train, ytype, atype, ex, ey, rho_x, rho_y, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_tf, y_train.ravel())
    y_pred = classifier.predict(X_test_tf)
    acc = 100 * np.sum(y_pred == y_test.ravel()) / y_pred.shape
    gsir_tup1 += (acc,)
    gsir_tup2 += (dc_gsir,)

    # The generalized sliced average variance estimation method.
    pred_test = gsave(X_train, X_test, y_train, ytype, ex, ey, rho_x, rho_y, args.latent_dim)
    dc_gsave = test_gsdr(pred_test, y_test, device)

    train_val_pred = gsave(X_train, X_train, y_train, ytype, ex, ey, rho_x, rho_y, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train_tf, y_train.ravel())
    y_pred = classifier.predict(X_test_tf)
    acc = 100 * np.sum(y_pred == y_test.ravel()) / y_pred.shape
    gsave_tup1 += (acc,)
    gsave_tup2 += (dc_gsave,)
print("done!")
print('latent_dim-', args.latent_dim)
print('GSIR ACC: {:.2f}({:.2f})\n'.format(np.mean(gsir_tup1), np.var(gsir_tup1) ** 0.5))
print('GSIR DC: {:.2f}({:.2f})\n'.format(np.mean(gsir_tup2), np.var(gsir_tup2) ** 0.5))
print(gsir_tup1, gsir_tup2)
print('GSAVE ACC: {:.2f}({:.2f})\n'.format(np.mean(gsave_tup1), np.var(gsave_tup1) ** 0.5))
print('GSAVE DC: {:.2f}({:.2f})\n'.format(np.mean(gsave_tup2), np.var(gsave_tup2) ** 0.5))
print(gsave_tup1, gsave_tup2)






