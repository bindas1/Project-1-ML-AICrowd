# -*- coding: utf-8 -*-

# File for all graded functions

import numpy as np

# Linear regression using gradient descent

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    
    Parameters
    ---------
    y - class labels vector
    tx - features array
    initial_w - initial weight
    max_iters - number of iterations
    gamma - step-size for the gradient descent
    
    Returns
    -------
    mse and optimal weights
    """

    losses, w = gradient_descent_mse(y, tx, initial_w, max_iters, gamma)

    return w[-1], losses[-1]
    

# Linear regression using stochastic gradient descent

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    calculate the least squares.
    
    Parameters
    ---------
    y - class labels vector
    tx - features array
    initial_w - initial weight
    max_iters - number of iterations
    gamma - step-size for the stochastic gradient descent
    
    Returns
    -------
    mse and optimal weights
    """
    
    losses, w = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)

    return w[-1], losses[-1]

# Least squares regression using normal equations

def least_squares(y, tx):
    """
    calculate the least squares.
    
    Parameters
    ---------
    y - class labels vector
    tx - features array
    
    Returns
    -------
    mse and optimal weights
    """
    
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

# Ridge regression using normal equations

def ridge_regression(y, tx, lambda_):
    """
    implement ridge regression.
    
    Parameters
    ---------
    y - class labels vector
    tx - features array
    lambda_ - tradeoff parameter
    
    Returns
    -------
    mse and optimal weights
    """
    
    N = tx.shape[0]
    lambda_prim = 2 * N * lambda_
    w = np.linalg.solve(tx.T@tx + lambda_prim * np.eye(tx.shape[1]), tx.T@y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

# Logistic regression using gradient descent or SGD

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform max iterations of logistic regression
    
    Parameters
    ---------
    y - class labels vector
    tx - features array
    initial_w - initial weight
    max_iters - number of iterations
    gamma - step-size for the gradient descent
    
    Returns
    -------
    loss and updated weights 
    """
    
    losses = []
    w = initial_w

    for i in range(max_iters):
        w, loss = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)

    return w, losses[-1]

# Regularized logistic regression using gradient descent or SGD

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform max iterations of logistic regression with regularization
    
    Parameters
    ---------
    y - class labels vector
    tx - features array
    lambda_ - tradeoff parameter
    initial_w - initial weight
    max_iters - number of iterations
    gamma - step-size for the gradient descent
    
    Returns
    -------
    loss and updated weights 
    """
    
    losses = []
    w = initial_w

    for i in range(max_iters):
        w, loss = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        if not (i % 10): 
            print('Iteration {}, loss {}'.format(i, loss))
        losses.append(loss)

    return w, losses[-1]



# Helper functions

# Helpers for cost computation

def calculate_mse(e):
    """Calculate the mse for error vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for error vector e."""
    return np.mean(np.abs(e))

def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse."""
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_loss_mae(y, tx, w):
    """Calculate the loss using mae."""
    e = y - tx.dot(w)
    return calculate_mae(e)

# Helpers for gradient descent

def compute_gradient_mse(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    
    grad = (-1/len(y)) * np.dot(tx.T, e)
    loss = compute_loss_mse(y, tx, w)
    
    return grad,loss

def compute_gradient_mae(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - np.dot(tx, w)
    
    grad = (-1/len(y)) * np.dot(tx.T, e)
    loss = compute_loss_mae(y, tx, w)
    
    return grad,loss

def gradient_descent_mse(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, loss = compute_gradient_mse(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws

def gradient_descent_mae(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, loss = compute_gradient_mae(y, tx, w)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad, e

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)

    return losses, ws

# Helpers for logistic regression

def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss_log(y, tx, w):
    """compute the loss: negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx @ w)) - y * (tx @ w))

def calculate_gradient_log(y, tx, w):
    """compute the gradient of loss."""
    return tx.T@(sigmoid(tx@w)-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss_log(y, tx, w)
    grad = calculate_gradient_log(y, tx, w)
    w = w - gamma * grad
    return w, loss

# Helpers for regularized logistic regression

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    h = sigmoid(tx@w)
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