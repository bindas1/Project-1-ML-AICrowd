# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np

from costs import *


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = tx.shape[0]
    lambda_prim = 2 * N * lambda_
    w = np.linalg.solve(tx.T@tx + lambda_prim * np.eye(tx.shape[1]), tx.T@y)
    loss = compute_loss_mse(y, tx, w)
    return w, loss
