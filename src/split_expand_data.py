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

def expand_X(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias but omitting interaction terms
    """
    
    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    return expand

def expand_X_terms(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias and interaction terms
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
    
    expand = expand_X(X,d)
    expand_withoutBias,mu,std = normalize(expand[:,1:])
    expand[:,1:] = expand_withoutBias
    return expand, mu, std


def split_data(X, y, split_ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    indices = np.arange(X.shape[0])
    n          = X.shape[0]
    X_train    = X[indices[0:int(n*split_ratio)],:]
    y_train    = y[indices[0:int(n*split_ratio)]] 
    X_test     = X[indices[int(n*(split_ratio)):],:] 
    y_test     = y[indices[int(n*(split_ratio)):]] 

    return X_train, y_train, X_test, y_test
