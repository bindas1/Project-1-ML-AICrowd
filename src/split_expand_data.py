# -*- coding: utf-8 -*-

import numpy as np

def normalize(X):
    """
    Normalize the data

    Returns the normalized data along with the mean and the standard deviation
    """
    mu    = np.mean(X,0,keepdims=True)
    std   = np.std(X,0,keepdims=True)
    X     = (X-mu)/std
    return X, mu, std

# same as build_poly(x, degree) from lab3
def expand_X(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias but omitting interaction terms
    """

    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    return expand

def expand_X_cross(X,d,cross_d):
    """
    perform degree-d polynomial feature expansion of X, with bias with some cross terms
    (up to degree cross_d).
    """

    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    for i in range(1, cross_d):
        print("enter")
        print(X)
        X_rotated = np.roll(X, i, axis=1)
        print(X_rotated)
        j = i-1
        n = 1
        while(j>0):
            print("enters while")
            X_rotated *= np.roll(X,i+n, axis=1)
            n += 1
            j-=1
        print(X_rotated)
        expand=np.hstack((expand, X*X_rotated))
        print(expand)
    return expand

def expand_X_cross_1(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias with every cross terms
    of degree 1.
    """

    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    for i in range(1, X.shape[1]):
        print("enter")
        X_rotated = np.roll(X, i, axis=1)
        print(X)
        print("STOP")
        print(X_rotated)
        expand=np.hstack((expand, X*X_rotated))
        print("Nothing")
        print(expand)
    return expand

def expand_X_trigo(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias but omitting cross terms,
    but with some trigonometric functions.
    """

    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    expand=np.hstack((expand, np.cos(X)))
    expand=np.hstack((expand, np.sin(X)))
    expand=np.hstack((expand, np.tan(X)))
    expand=np.hstack((expand, np.arctan(X)))
    return expand


def expand_X_cross_1_trigo(X,d,cross_d):
    """
    perform degree-d polynomial feature expansion of X, with bias with some cross terms
    (up to degree cross_d).
    """

    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    for i in range(1, cross_d):
        print("enter")
        print(X)
        X_rotated = np.roll(X, i, axis=1)
        print(X_rotated)
        j = i-1
        n = 1
        while(j>0):
            print("enters while")
            X_rotated *= np.roll(X,i+n, axis=1)
            n += 1
            j-=1
        print(X_rotated)
        expand=np.hstack((expand, X*X_rotated))
        print(expand)
    expand=np.hstack((expand, np.cos(X)))
    expand=np.hstack((expand, np.sin(X)))
    expand=np.hstack((expand, np.tan(X)))
    expand=np.hstack((expand, np.arctan(X)))
    return expand


def expand_X_terms(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias and interaction terms.
    """

    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1):
        expand = np.hstack((expand, X**idx))
        for feature1 in range(0, X.shape[1]):
            for feature2 in range(0, feature1):
                for p in range(1, idx):
                    new_tab = np.multiply(X[:,feature1]**p,X[:,feature2]**(idx-p))
                    expand = np.concatenate((expand, np.multiply(X[:,feature1]**p,X[:,feature2]**(idx-p)).reshape(X.shape[0],1)), axis=1)
    return expand

def expand_and_normalize_X(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias but omitting interaction terms
    and normalize them.
    """

    #expand = expand_X(X,d)
    expand = expand_X_cross_1_trigo(X, d, 2)
    expand_withoutBias,mu,std = normalize(expand[:,1:])
    expand[:,1:] = expand_withoutBias
    return expand, mu, std


def split_data(X, y, split_ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)

    n = X.shape[0]
    indices = indices = np.random.permutation(n)

    ind_split = int(n*split_ratio)

    X_train    = X[indices[0:ind_split],:]
    y_train    = y[indices[0:ind_split]]
    X_test     = X[indices[ind_split:],:]
    y_test     = y[indices[ind_split:]]

    return X_train, y_train, X_test, y_test
