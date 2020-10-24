# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx @ w)) - y * (tx @ w))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * grad
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
	"""
	Perform max iterations of logistic regression
	Return loss and updated weights
	"""
	losses = []
	w = initial_w

	for i in range(max_iters):
		w, loss = learning_by_gradient_descent(y, tx, w, gamma)
		# if not (i % 10): 
		# 	print('Iteration {}, loss {}'.format(i, loss))
		losses.append(loss)

	return w, losses[-1]
