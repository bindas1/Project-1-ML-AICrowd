# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import *


def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    # returns mse, and optimal weights
    # ***************************************************
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss

