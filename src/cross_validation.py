# -*- coding: utf-8 -*-
# plots and functions for cross-validation and bias variance distributions

import numpy as np
import matplotlib.pyplot as plt

from costs import compute_loss_mse
from least_squares import *
from implementations import *
from split_expand_data import *
from proj1_helpers import *


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_test = y[te_indice]
    y_train = y[tr_indice]
    x_test = x[te_indice]
    x_train = x[tr_indice]
    
    x_train_poly, mu_train_poly, std_train_poly = expand_and_normalize_X(x_train,degree)
    x_test_poly  = expand_X(x_test,degree)
    x_test_poly[:,1:]  = (x_test_poly[:,1:]-mu_train_poly)/std_train_poly
    
    w = ridge_regression(y_train, x_train_poly, lambda_)[0]
    
    loss_tr = compute_loss_mse(y_train, x_train_poly, w)
    loss_te = compute_loss_mse(y_test, x_test_poly, w)
    return loss_tr, loss_te, w

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("../img/cross_val_ridge.png")
    plt.yscale("log")

def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig("../img/bias_variance.png")
    plt.legend()

def bias_variance_demo(y, tX_clean):
    """The entry."""
    # define parameters
    seeds = range(6)
    ratio_train = 0.7
    degrees = range(1, 10)
    
    # define list to store the variable
    mse_tr = np.empty((len(seeds), len(degrees)))
    mse_te = np.empty((len(seeds), len(degrees)))
    
    for index_seed, seed in enumerate(seeds):
        x_train, y_train, x_test, y_test = split_data(tX_clean, y, split_ratio=ratio_train, seed=seed)
        for i, d in enumerate(degrees):
            X_train_poly, mu_train_poly, std_train_poly = expand_and_normalize_X(X_train,d)
            X_test_poly  = expand_X(X_test,d)
            X_test_poly[:,1:]  = (X_test_poly[:,1:]-mu_train_poly)/std_train_poly
            w = least_squares(y_train, X_train_poly)[0]
            mse_tr[index_seed, i] = compute_loss_mse(y_train, X_train_poly, w)
            mse_te[index_seed, i] = compute_loss_mse(y_test, X_test_poly, w)

    bias_variance_decomposition_visualization(degrees, mse_tr, mse_te)

def cross_validation_demo(y, tX_clean):
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-6, -1, 15)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    for lambda_ in lambdas:
        tr_tmp = []
        te_tmp = []
        for k in range(k_fold):
            tr, te, _ = cross_validation(y, tX_clean, k_indices, k, lambda_, degree)
            tr_tmp.append(tr)
            te_tmp.append(te)
        mse_tr.append(np.mean(tr_tmp))
        mse_te.append(np.mean(te_tmp))
    cross_validation_visualization(lambdas, mse_tr, mse_te)

# y, tX_clean = load_clean_data()

# run this to achieve cross validation for degree 7, this may take a while (4 folds)
# cross_validation_demo(y, tX_clean)

# run this to obtain bias variance plot
# bias_variance_demo(y, tX_clean)

