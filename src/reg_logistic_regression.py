# -*- coding: utf-8 -*-
# Logistic regression with regularization using hessian

import numpy as np

from logistic_regression import *

# both of these  to kernel dead, first one is my code, dunno whats wrong
def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
#     S = sigmoid(tx @ w) * (1 - sigmoid(tx @ w))
#     S_diag = np.diag(S[:])
#     print(S_diag.shape)
#     return tx.T @ S_diag @ tx
    h = sigmoid(tx@w)
    print(h.shape)
    h = np.diag(h.reshape(-1,1).T[0])
    r = np.multiply(h, (1-h))
    return tx.T@r@tx

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient and hessian"""
    loss = calculate_loss_log(y, tx, w) + lambda_ * (np.linalg.norm(w, 2) ** 2) / 2
    grad = calculate_gradient_log(y, tx, w)
    hess = calculate_hessian(y, tx, w) + lambda_
    return loss, grad, hess

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad, hess = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * np.linalg.inv(hess) @ grad
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform max iterations of logistic regression with regularization
    Return loss and updated weights
    """
    losses = []
    w = initial_w

    for i in range(max_iters):
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if not (i % 10): 
            print('Iteration {}, loss {}'.format(i, loss))
        losses.append(loss)

    return w, losses[-1]
    

# Code that I was running in jupyter notebook

y_train_log = y_train
y_test_log = y_test

y_train_log[y_train == -1] = 0
y_test_log[y_test == -1] = 0


max_iters = 100
initial_w = np.random.normal(0, 1e-4, X_train_poly.shape[1])
gamma = 0.01
lambda_ = 0.1

w, loss = reg_logistic_regression(y_train_log, X_train_poly, lambda_, initial_w, max_iters, gamma)
 
