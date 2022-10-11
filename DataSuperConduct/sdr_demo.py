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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import argparse
import os

# arguments setting
parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, default=5, help='the dimension of latent space')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'sim/toy_reg'
setup_seed(args.seed, args.cuda)

print('latent_dim-', args.latent_dim, 'cuda', args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

sir_tup1 = ()
sir_tup2 = ()

save_tup1 = ()
save_tup2 = ()

# The parameter for cross-validation to select the best hyper-parameters for the linear SDR methods.
slice_seq = np.array([5, 10, 15, 20, 25, 30])
ytype = 'scalar'

# The splitting number of cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=421)
mydata = np.loadtxt('./superconduct/train.csv', delimiter=',', skiprows=1)
X = mydata[:, :-1]
y = mydata[:, -1].reshape((-1, 1))
X = X.astype(np.float32) # Covariate data
y = y.astype(np.float32) # Response data
validation_n = 3000 # Validation sample size

for train_val_index, test_index in kf.split(X):
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    ind = np.arange(train_val_index.shape[0])
    train_n = train_val_index.shape[0] - validation_n
    test_n = test_index.shape[0]
    np.random.shuffle(ind)
    X_train, X_validation = X_train_val[ind[:train_n]], X_train_val[ind[train_n:]]
    y_train, y_validation = y_train_val[ind[:train_n]], y_train_val[ind[train_n:]]

    pre_scaler = StandardScaler()
    pre_scaler.fit(X_train)
    X_train = pre_scaler.transform(X_train)
    X_validation = pre_scaler.transform(X_validation)
    X_test = pre_scaler.transform(X_test)

    # The sliced inverse regression method.
    opt_sn_sir = sdr_slice_select(sir_sdr, X_train, y_train, X_validation, y_validation, args.latent_dim, slice_seq,
                                  device)
    pred_test = sir_sdr(X_train, X_test, y_train, opt_sn_sir, ytype, args.latent_dim)
    dc_sir = test_gsdr(pred_test, y_test, device)

    train_val_pred = sir_sdr(X_train, X_train, y_train, opt_sn_sir, ytype, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    reg = LinearRegression().fit(X_train_tf, y_train)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    sir_tup1 += (dc_sir,)
    sir_tup2 += (rms,)

    # The sliced average variance estimation method.
    opt_sn_save = sdr_slice_select(save_sdr, X_train, y_train, X_validation, y_validation, args.latent_dim, slice_seq,
                                   device)
    pred_test = save_sdr(X_train, X_test, y_train, opt_sn_save, ytype, args.latent_dim)
    dc_save = test_gsdr(pred_test, y_test, device)

    train_val_pred = save_sdr(X_train, X_train, y_train, opt_sn_save, ytype, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    reg = LinearRegression().fit(X_train_tf, y_train)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    save_tup1 += (dc_save,)
    save_tup2 += (rms,)
print("done!")
print('latent_dim-', args.latent_dim)
print('SIR DC \tmean: {:.2f}({:.2f})\n'.format(np.mean(sir_tup1), np.var(sir_tup1) ** 0.5))
print('SIR MSE \tmean: {:.2f}({:.2f})\n'.format(np.mean(sir_tup2), np.var(sir_tup2) ** 0.5))
print(sir_tup1)
print(sir_tup2)
print('SAVE DC \tmean: {:.2f}({:.2f})\n'.format(np.mean(save_tup1), np.var(save_tup1) ** 0.5))
print('SAVE MSE \tmean: {:.2f}({:.2f})\n'.format(np.mean(save_tup2), np.var(save_tup2) ** 0.5))
print(save_tup1)
print(save_tup2)
