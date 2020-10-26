# -*- coding: utf-8 -*-
# Logistic regression with regularization using hessian

import numpy as np

from logistic_regression import *


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_loss_log(y, tx, w) + lambda_ * (np.linalg.norm(w, 2) ** 2) / 2
    grad = calculate_gradient_log(y, tx, w)
    return loss, grad

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
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
        losses.append(loss)

    return w, losses[-1]
 
