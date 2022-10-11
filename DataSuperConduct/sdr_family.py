# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 23:25
# @Author  : SimonZHENG
# @Email   : zhengsiming2016@163.com
# @File    : sdr_family.py
# @Software: PyCharm
import math
from scipy.special import comb
import numpy as np
from numpy.linalg import matrix_power as matpower
from sklearn.model_selection import KFold
from model_nets import cor
import torch
import random


def discretize_y(y, sliced_h):
    """
    Discretize the np.array y.
    Args:
        y (np.array): An n*1 data matrices of the response stored in numpy array data type.
        sliced_h (int): The slice number.
    """
    n = len(y)
    m = math.floor(n/sliced_h)
    y = y + 0.00001 * np.mean(y) * np.random.randn(n, 1)
    ord_y = np.sort(y, 0)
    divpt = []
    for i in range(1, sliced_h):
        divpt.append(ord_y[i * m])
    divpt = np.array(divpt)
    y1 = np.zeros((n, 1))
    y1[y < divpt[0]] = 1
    y1[y >= divpt[sliced_h-2]] = sliced_h
    for i in range(sliced_h-2):
        y1[(divpt[i] <= y) & (y < divpt[i+1])] = i+2
    return y1


def gram_gauss(x, x_new, rho_x):
    """
    Calculate the Gaussian gram matrix between x and x_new.
    Args:
        x, x_new (np.array): The n*p and m*p data matrices of the covariates stored in numpy array data type.
        rho_x (float): The bandwidth.
    """
    n = x.shape[0]
    n_new = x_new.shape[0]
    k2 = x @ x.T
    k1 = np.ones((n, 1)) @ np.mat(np.diag(k2))
    k = k1 - 2 * k2 + k1.T
    squared_sigma = (np.sum(k) + np.sum(np.diag(k2))) / (2 * comb(n, 2))
    gamma = rho_x / (2 * squared_sigma)
    k1_new = np.transpose(np.ones((n_new, 1)) @ np.mat(np.diag(k2)))
    k2_new = x @ x_new.T
    k3_new = np.ones((n, 1)) @ np.mat(np.diag(x_new @ x_new.T))
    return np.exp(-1 * gamma * (k1_new - 2 * k2_new + k3_new))


def gram_dis(y):
    """
    Calculate the gram matrix for the discrete y.
    Args:
        y (np.array): An n*1 data matrices of the response stored in numpy array data type.
    """
    n = len(y)
    yy = y @ np.ones((1, n))
    diff = yy - np.transpose(yy)
    vecker = np.zeros((n, n))
    vecker[diff == 0] = 1
    return vecker


def onorm(a):
    """
    Orthogonalize the square matrix a.
    Args:
        a (np.array): An n*n matrix in numpy array data type.
    """
    _, s, _ = np.linalg.svd((a + a.T) / 2)
    return np.max(s)


def mat_sym(a):
    """
    symmetrize the square matrix a.
    Args:
        a (np.array): An n*n matrix in numpy array data type.
    """
    return (a + a.T) / 2


def matpower2(a, alpha):
    """
    Calculate the matrix a to the power of alpha.
    Args:
        a (np.array): An n*n matrix in numpy array data type.
        alpha (float): The power degree.
    """
    a = (a + a.T) / 2
    u, s, vt = np.linalg.svd(a)
    return u @ np.diag(s ** alpha) @ vt


def ridgepower(a, e: float, c: float):
    return matpower2(a + e * onorm(a) * np.eye(a.shape[0]), c)


def gsir(x, x_new, y, y_type, a_type, ex, ey, rho_x, rho_y, latent_d):
    """
    The generalized sliced inverse regression method.
    Args:
        x, x_new (np.array): The n*p and m*p data matrices of the covariates stored in numpy array data type.
        y (np.array): An n*1 data matrices of the response stored in numpy array data type.
        ex, ey, rho_x, rho_y (float): The four hyperparameters required for the scalar response and ey, rho_y
                                      are not need when the response is categorical.
        ytype (string): The type of data y: scalar or categorical.
        a_type (np.array): The square matrix required.
        latent_d (int): The dimensionality of the desired CDR space.
    """
    n = x.shape[0]
    mat_q = np.eye(n) - np.ones((n, n)) / n
    kx = gram_gauss(x, x, rho_x)
    if y_type == 'scalar':
        ky = gram_gauss(y, y, rho_y)
    elif y_type == 'categorical':
        ky = gram_dis(y)
    mat_gx = mat_q @ kx @ mat_q
    mat_gy = mat_q @ ky @ mat_q
    inv_mat_gx = matpower(mat_sym(mat_gx + ex * onorm(mat_gx) * np.eye(n)), -1)
    if y_type == 'scalar':
        inv_mat_gy = matpower(mat_sym(mat_gy + ey * onorm(mat_gy) * np.eye(n)), -1)
    elif y_type == 'categorical':
        inv_mat_gy = np.linalg.pinv(mat_sym(mat_gy), 1e-9)
    a1 = inv_mat_gx @ mat_gx
    if a_type == 'identity':
        a2 = mat_gy
    elif a_type == 'Gyinv':
        a2 = mat_gy @ inv_mat_gy
    gsir_mat = a1 @ a2 @ a1.T
    u, *_ = np.linalg.svd(mat_sym(gsir_mat))
    v = u[:, 0:latent_d]
    kx_new = gram_gauss(x, x_new, rho_x)
    pred_new = np.transpose(v.T @ inv_mat_gx @ mat_q @ kx_new)
    return pred_new


def gsave(x, x_new, y, y_type, ex, ey, rho_x, rho_y, latent_d):
    """
    The generalized sliced average variance estimation method.
    Args:
        x, x_new (np.array): The n*p and m*p data matrices of the covariates stored in numpy array data type.
        y (np.array): An n*1 data matrices of the response stored in numpy array data type.
        ex, ey, rho_x, rho_y: The four hyperparameters required for the scalar response and ey, rho_y are not need
                              when the response is categorical.
        ytype (string): The type of data y: scalar or categorical.
        latent_d (int): The dimensionality of the desired CDR space.
    """
    n = x.shape[0]
    kx0 = gram_gauss(x, x, rho_x)
    kx = np.concatenate((np.ones((1, n)), kx0), axis=0)
    if y_type == 'scalar':
        ky0 = gram_gauss(y, y, rho_y)
    elif y_type == 'categorical':
        ky0 = gram_dis(y)
    ky = np.concatenate((np.ones((1, n)), ky0), axis=0)
    mat_q = np.eye(n) - np.ones((n, n)) / n
    kky = ky @ ky.T
    if y_type == 'scalar':
        kky_inv = ridgepower(mat_sym(kky), ey, -1)
    elif y_type == 'categorical':
        kky_inv = np.linalg.pinv(mat_sym(kky), 1e-9)
    piy = ky.T @ kky_inv @ ky
    sumlam = np.diag(np.sum(piy, 1)) - piy @ piy
    a1 = np.diag(np.sum(piy * piy, 1)) - piy @ piy / n
    tem_mat = np.diag(np.mean(np.array(piy), 1))
    a2 = (piy * piy) @ piy - piy @ tem_mat @ piy
    a3 = piy @ np.diag(np.diag(piy @ mat_q @ piy)) @ piy
    mid = mat_q / n - (2 / n) * mat_q @ sumlam @ mat_q + mat_q @ (a1 - a2 - a2.T + a3) @ mat_q
    kx_new0 = gram_gauss(x, x_new, rho_x)
    n1 = kx_new0.shape[1]
    kx_new = np.concatenate((np.ones((1, n1)), kx_new0), axis=0)
    mat_q1 = np.eye(n1) - np.ones((n1, n1)) / n1
    kk = ridgepower(kx @ mat_q @ kx.T, ex, -1 / 2) @ kx @ mat_q
    kk_new = ridgepower(kx @ mat_q @ kx.T, ex, -1 / 2) @ kx_new @ mat_q1
    u, *_ = np.linalg.svd(mat_sym(kk @ mid @ kk.T))
    v = u[:, 0:latent_d]
    pred_y = kk_new.T @ v
    return pred_y


def test_gsdr(pred_new, y_test, device):
    """
    Calculate the empirical distance correlation between the learned features and the response.
    Args:
        pred_new (np.array): An n*p data matrix of the learned features stored in numpy array data type.
        y_test (np.array): An n*1 data matrix of the response stored in numpy array data type.
        device: Computation device: CPU or GPU
    """
    n = pred_new.shape[0]
    pred_new_t = torch.from_numpy(pred_new).float()
    y_test_t = torch.from_numpy(y_test).float()
    dcov = cor(pred_new_t.to(device), y_test_t.to(device), n, device)
    return dcov.cpu().numpy().item()


def cv_gsdr_para_select(x, y, folds, ep_seq, rho_seq):
    """
    Cross-validation to select the best hyperparameters for the generalized SDR method.
    Args:
        x (np.array): An n*p data matrix of the covariates stored in numpy array data type.
        y (np.array): An n*1 data matrix of the response stored in numpy array data type.
        folds (int): The number of splitting folds.
        ep_seq (list): The list of all candidate epsilons.
        rho_seq (list): The list of all candidate rhos.
    """
    temp = None
    ep_index = None
    rho_index = None
    kf = KFold(n_splits=folds)
    for i, ep in enumerate(ep_seq):
        for j, rho in enumerate(rho_seq):
            cv_error = 0
            for train_index, val_index in kf.split(x):
                x_tra, x_tes = x[train_index], x[val_index]
                y_tra, y_tes = y[train_index], y[val_index]
                n = y_tra.shape[0]
                kx = gram_gauss(x_tra, x_tra, rho)
                kx_tes = gram_gauss(x_tra, x_tes, rho)
                ky = gram_gauss(y_tra, y_tra, 1)
                ky_tes = gram_gauss(y_tra, y_tes, 1)
                mat = ky_tes.T - kx_tes.T @ matpower(kx + ep * onorm(kx) * np.eye(n), -1) @ ky
                cv_error = cv_error + np.linalg.norm(mat)
            if temp is None or cv_error < temp:
                temp = cv_error
                ep_index = i
                rho_index = j
    return ep_seq[ep_index], rho_seq[rho_index]


def sir_sdr(x, x_new, y, sliced_h, ytype, latent_d):
    """
    The sliced inverse regression method.
    Args:
        x, x_new (np.array): The n*p and m*p data matrices of the covariates stored in numpy array data type.
        y (np.array): An n*1 data matrices of the response stored in numpy array data type.
        slice_h (int): The slice number.
        ytype (string): The type of data y: scalar or categorical
        latent_d (int): The dimensionality of the desired CDR space.
    """
    n = x.shape[0]
    p = x.shape[1]
    signrt = matpower2(np.cov(x.T), -1/2)
    xst = (x - np.ones((n, 1)) @ np.mat(x.mean(0))) @ signrt
    if ytype == 'scalar':
        ydis = discretize_y(y, sliced_h)
    elif ytype == 'categorical':
        ydis = y
    ylabel = list(set(ydis.flatten().tolist()))
    slicenum = len(ylabel)
    prob = np.zeros(slicenum)
    exy = np.zeros((slicenum, p))
    for i in range(slicenum):
        prob[i] = len(ydis[ydis == ylabel[i]])/n
        exy[i, :] = xst[np.squeeze(ydis) == ylabel[i], :].mean(0)
    sirmat = exy.T @ np.diag(prob) @ exy
    u, *_ = np.linalg.svd(sirmat)
    v = u[:, 0:latent_d]
    n_new = x_new.shape[0]
    xst = (x_new - np.ones((n_new, 1)) @ np.mat(x.mean(0))) @ signrt
    pred = xst @ v
    return pred


def save_sdr(x, x_new, y, sliced_h, ytype, latent_d):
    """
    The sliced average variance estimation method.
    Args:
        x, x_new (np.array): The n*p and m*p data matrices of the covariates stored in numpy array data type.
        y (np.array): An n*1 data matrices of the response stored in numpy array data type.
        slice_h (int): The slice number.
        ytype (string): The type of data y: scalar or categorical
        latent_d (int): The dimensionality of the desired CDR space.
    """
    n = x.shape[0]
    p = x.shape[1]
    signrt = matpower2(np.cov(x.T), -1/2)
    xst = (x - np.ones((n,1)) @ np.mat(x.mean(0))) @ signrt
    if ytype == 'scalar':
        ydis = discretize_y(y, sliced_h)
    elif ytype == 'categorical':
        ydis = y
    ylabel = list(set(ydis.flatten().tolist()))
    slicenum = len(ylabel)
    savemat = np.zeros((p, p))
    ide_mat = np.eye(p)
    for i in range(slicenum):
        prob = len(ydis[ydis == ylabel[i]])/n
        vxy = np.cov(xst[np.squeeze(ydis) == ylabel[i], :].T)
        savemat = savemat + prob * (vxy - ide_mat) @ (vxy - ide_mat)
    u, *_ = np.linalg.svd(savemat)
    v = u[:, 0:latent_d]
    n_new = x_new.shape[0]
    xst = (x_new - np.ones((n_new, 1)) @ np.mat(x.mean(0))) @ signrt
    pred = xst @ v
    return pred


def sdr_slice_select(sdr_method, x_tra, y_tra, x_tes, y_tes, latent_d, slice_seq, device):
    """
    Cross-validation based on empirical DC to select the best slice number.
    Args:
        sdr_method (function): The targeted SDR method: sir_sdr() or save_sdr.
        x_tra, x_tes (np.array): Two n*p data matrices of the covariates stored in numpy array data type.
        y_tra, y_tes (np.array): Two n*1 data matrices of the response stored in numpy array data type.
        latent_d (int): The dimensionality of the desired CDR space.
        slice_seq (list): The list of all candidate slice number.
        device: Computation device: CPU or GPU
    """
    ytype = 'scalar'
    temp = None
    slice_index = None
    for i, sliced_h in enumerate(slice_seq):
        pred_test = sdr_method(x_tra, x_tes, y_tra, sliced_h, ytype, latent_d)
        dc = test_gsdr(pred_test, y_tes, device)
        if temp is None or dc > temp:
            temp = dc
            slice_index = i
    return slice_seq[slice_index]


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
