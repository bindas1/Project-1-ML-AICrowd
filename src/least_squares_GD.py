# -*- coding: utf-8 -*-

import numpy as np

from gradient_descent import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent"""
    # ***************************************************
    # returns mse, and optimal weights
    # ***************************************************

    losses, w = gradient_descent_mse(y, tx, initial_w, max_iters, gamma)

    return w[-1], losses[-1]

