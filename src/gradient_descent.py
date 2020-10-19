import numpy as np

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
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

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
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws




