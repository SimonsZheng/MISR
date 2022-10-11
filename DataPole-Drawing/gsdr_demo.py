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
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()
print('latent_dim-', args.latent_dim)

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.save = args.save or 'sim/toy_reg'
setup_seed(args.seed, args.cuda)

if not os.path.exists(args.save):
    os.makedirs(args.save, exist_ok=True)

# The parameter for cross-validation to select the best hyper-parameters for the generalized SDR methods.
folds_num = 4
ep_seq = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
rho_seq = np.array([1 / 5, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 5])
ytype = 'scalar'
atype = 'identity'

poldata = np.loadtxt('./pol/PoleTelecomm/poldata.csv', delimiter=',')
poltest = np.loadtxt('./pol/PoleTelecomm/poltest.csv', delimiter=',')
mydata = np.concatenate((poldata, poltest))
X_old = mydata[:, :-1]
y = mydata[:, -1].reshape((-1, 1))
X_old = X_old.astype(np.float32)
y = y.astype(np.float32)

# Delete the non-informative features
Inf_col = np.zeros(X_old.shape[1])
for i in range(X_old.shape[1]):
    l = X_old[:, i]
    Inf_col[i] = len(list(set(l.flatten().tolist())))
X = X_old[:, Inf_col != 1]

train_n = 10000 # Training sample size
validation_n = 2000 # Validation sample size
test_n = 3000 # Testing sample size
n_sample = train_n + validation_n + test_n

ind = np.arange(n_sample)
np.random.shuffle(ind)
print(ind[:5])
X_train_val_pre, X_test_pre = X[ind[:train_n + validation_n],], X[ind[train_n + validation_n:],]
y_train_val_pre, y_test_pre = y[ind[:train_n + validation_n],], y[ind[train_n + validation_n:],]

gtrain_n = 4000 # Training subsample size
gvalidation_n = 2000 # Validation sample size
gtest_n = 3000 # Testing sample size

X_train_val, X_test = X_train_val_pre[:gvalidation_n + gtrain_n, ], X_test_pre[:gtest_n, ]
y_train_val, y_test = y_train_val_pre[:gvalidation_n + gtrain_n, ], y_test_pre[:gtest_n, ]
ind = np.arange(gtrain_n + gvalidation_n)
np.random.shuffle(ind)
X_train, X_validation = X_train_val[ind[:gtrain_n]], X_train_val[ind[gtrain_n:]]
y_train, y_validation = y_train_val[ind[:gtrain_n]], y_train_val[ind[gtrain_n:]]

pre_scaler = StandardScaler()
pre_scaler.fit(X_train)
X_train = pre_scaler.transform(X_train)
X_validation = pre_scaler.transform(X_validation)
X_test = pre_scaler.transform(X_test)

# Cross-validation to select the best hyper-parameters for the generalized SDR methods.
ex, rho_x = cv_gsdr_para_select(X_validation, y_validation, folds_num, ep_seq, rho_seq)
ey, rho_y = cv_gsdr_para_select(y_validation, X_validation, folds_num, ep_seq, rho_seq)

# The generalized sliced inverse regression method.
pred_test = gsir(X_train, X_test, y_train, ytype, atype, ex, ey, rho_x, rho_y, args.latent_dim)
saved_data = np.concatenate((pred_test, y_test), axis=1)
np.savetxt('GSIR.csv', saved_data, delimiter=',')

# The generalized sliced average variance estimation method.
pred_test = gsave(X_train, X_test, y_train, ytype, ex, ey, rho_x, rho_y, args.latent_dim)
saved_data = np.concatenate((pred_test, y_test), axis=1)
np.savetxt('GSAVE.csv', saved_data, delimiter=',')
