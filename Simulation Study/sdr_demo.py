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
parser.add_argument('--latent_dim', type=int, default=2, help='the dimension of latent space')
parser.add_argument('--cuda_device', type=int, default=0, help='choose cuda device')
parser.add_argument('--no-cuda', action='store_true', help='if TRUE, cuda will not be used')
parser.add_argument('--save', help='path to save results')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--model', type=int, default=2, help='1: Model A; 2: Model B; 3: Model C')
parser.add_argument('--scenario', type=int, default=2, help='1: scenario 1 ; 2: scenario 2; '
                                                            '3: scenario 3; 4: scenario 4')
args = parser.parse_args()
print('Model-', args.model, 'Scenario-', args.scenario)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'sim/toy_reg'
setup_seed(args.seed, args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

# Training sample size
train_n = 4000
# Validation sample size
validation_n = 1000
# Testing sample size
test_n = 1000

gsir_tup1 = ()
gsir_tup2 = ()

gsave_tup1 = ()
gsave_tup2 = ()

sir_tup1 = ()
sir_tup2 = ()

save_tup1 = ()
save_tup2 = ()

# The parameter for cross-validation to select the best hyperparameters for the SDR methods.
folds_num = 4
ep_seq = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
rho_seq = np.array([1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5])
slice_seq = np.array([5, 10, 15, 20, 25, 30])
ytype = 'scalar'
atype = 'identity'

# The splitting number of cross-validation
kf = KFold(n_splits=6)
data_path = args.save+'/SimData'
SavedData = np.loadtxt('./'+data_path+'/Data_S('+str(args.scenario)+')_M('+str(args.model)+').csv',
                       delimiter=',')
X = SavedData[:, :-1]
y = SavedData[:, -1].reshape((-1, 1))
for train_val_index, test_index in kf.split(X):
    X_train_val, X_test = X[train_val_index], X[test_index]
    y_train_val, y_test = y[train_val_index], y[test_index]
    ind = np.arange(train_n + validation_n)
    np.random.shuffle(ind)
    X_train, X_validation = X_train_val[ind[:train_n]], X_train_val[ind[train_n:]]
    y_train, y_validation = y_train_val[ind[:train_n]], y_train_val[ind[train_n:]]

    # Cross-validation to select the best hyper-parameters for the generalized SDR methods.
    ex, rho_x = cv_gsdr_para_select(X_validation, y_validation, folds_num, ep_seq, rho_seq)
    ey, rho_y = cv_gsdr_para_select(y_validation, X_validation, folds_num, ep_seq, rho_seq)

    # The generalized sliced inverse regression method.
    pred_test = gsir(X_train, X_test, y_train, ytype, atype, ex, ey, rho_x, rho_y, args.latent_dim)
    dc_gsir = test_gsdr(pred_test, y_test, device)

    train_val_pred = gsir(X_train, X_train_val, y_train, ytype, atype, ex, ey, rho_x, rho_y, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    reg = LinearRegression().fit(X_train_tf, y_train_val)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    gsir_tup1 += (dc_gsir,)
    gsir_tup2 += (rms,)

    pred_test = gsave(X_train, X_test, y_train, ytype, ex, ey, rho_x, rho_y, args.latent_dim)
    dc_gsave = test_gsdr(pred_test, y_test, device)

    # The generalized sliced average variance estimation method.
    train_val_pred = gsave(X_train, X_train_val, y_train, ytype, ex, ey, rho_x, rho_y, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    reg = LinearRegression().fit(X_train_tf, y_train_val)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    gsave_tup1 += (dc_gsave,)
    gsave_tup2 += (rms,)

    # The sliced inverse regression method.
    opt_sn_sir = sdr_slice_select(sir_sdr, X_train, y_train, X_validation, y_validation, args.latent_dim, slice_seq,
                                  device)
    pred_test = sir_sdr(X_train, X_test, y_train, opt_sn_sir, ytype, args.latent_dim)
    dc_sir = test_gsdr(pred_test, y_test, device)

    train_val_pred = sir_sdr(X_train, X_train_val, y_train, opt_sn_sir, ytype, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    reg = LinearRegression().fit(X_train_tf, y_train_val)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    sir_tup1 += (dc_sir,)
    sir_tup2 += (rms,)

    # The sliced average variance estimation method.
    opt_sn_save = sdr_slice_select(save_sdr, X_train, y_train, X_validation, y_validation, args.latent_dim, slice_seq,
                                   device)
    pred_test = save_sdr(X_train, X_test, y_train, opt_sn_save, ytype, args.latent_dim)
    dc_save = test_gsdr(pred_test, y_test, device)

    train_val_pred = save_sdr(X_train, X_train_val, y_train, opt_sn_save, ytype, args.latent_dim)
    scaler = StandardScaler()
    scaler.fit(train_val_pred)
    X_train_tf = scaler.transform(train_val_pred)
    X_test_tf = scaler.transform(pred_test)
    reg = LinearRegression().fit(X_train_tf, y_train_val)
    y_pred = reg.predict(X_test_tf)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    save_tup1 += (dc_save,)
    save_tup2 += (rms,)

print("done!")
print('SIR DC \tmean: {:.2f}({:.2f})\n'.format(np.mean(sir_tup1), np.var(sir_tup1) ** 0.5))
print('SIR MSE \tmean: {:.2f}({:.2f})\n'.format(np.mean(sir_tup2), np.var(sir_tup2) ** 0.5))

print('SAVE DC \tmean: {:.2f}({:.2f})\n'.format(np.mean(save_tup1), np.var(save_tup1) ** 0.5))
print('SAVE MSE \tmean: {:.2f}({:.2f})\n'.format(np.mean(save_tup2), np.var(save_tup2) ** 0.5))

print('GSIR DC \tmean: {:.2f}({:.2f})\n'.format(np.mean(gsir_tup1), np.var(gsir_tup1) ** 0.5))
print('GSIR MSE \tmean: {:.2f}({:.2f})\n'.format(np.mean(gsir_tup2), np.var(gsir_tup2) ** 0.5))

print('GSAVE DC \tmean: {:.2f}({:.2f})\n'.format(np.mean(gsave_tup1), np.var(gsave_tup1) ** 0.5))
print('GSAVE MSE \tmean: {:.2f}({:.2f})\n'.format(np.mean(gsave_tup2), np.var(gsave_tup2) ** 0.5))
