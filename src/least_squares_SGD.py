# -*- coding: utf-8 -*-

import numpy as np

from gradient_descent import *


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """calculate the least squares."""
    # ***************************************************
    # returns mse, and optimal weights
    # ***************************************************
    losses, w = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma)

    return w[-1], losses[-1]

