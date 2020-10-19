# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_loss_mse(y, tx, w):
    """Calculate the loss using mse."""
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_loss_mae(y, tx, w):
    """Calculate the loss using mae."""
    e = y - tx.dot(w)
    return calculate_mae(e)

