# -*- coding: utf-8 -*-
"""File to obtain the best results that we obtained on AI Crowd."""

from implementations import *
from proj1_helpers import *
from split_expand_data import *


y, tX_clean = load_clean_data()

# split data into train and test sets
X_train, y_train, X_test, y_test = split_data(tX_clean, y, split_ratio=0.8)

# store degree to later use with real test set
degree=8

# create expanded X_train and X_test and normalize
X_train_poly, mu_train_poly, std_train_poly = expand_and_normalize_X(X_train,degree)
X_test_poly  = expand_X(X_test,degree)
X_test_poly[:,1:]  = (X_test_poly[:,1:]-mu_train_poly)/std_train_poly

# least squares solution
w, loss = least_squares(y_train, X_train_poly)

# obtain final predictions



