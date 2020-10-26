# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

BOUND_DELETE=0.5
BOUND_CHANGE=0.05


def clean_data(X, bound_delete=BOUND_DELETE, bound_change=BOUND_CHANGE):
    """remove columns where standard deviation is 0

    Returns
    -------
    X : array with removed features
    """
    ind_delete = []

    # number of rows
    no_rows = X.shape[0]

    for i in range(1, X.shape[1]):
        if(np.std(X[:, i]) == 0):
            print("Column :", i)
            ind_delete.append(i)

    X = np.delete(X, ind_delete, axis=1)
    return X

def clean_data_999(X, bound_delete=BOUND_DELETE, bound_change=BOUND_CHANGE):
    """remove columns where values are -999 (the frequence should be either 0 or 1, except for column 0 which is not considered here
    
        Returns
        -------
        X : array with removed features
    """
    ind_delete = []
    
    # number of rows
    no_rows = X.shape[0]
    
    for i in range(1, X.shape[1]):
        freq = X[X[:, i]== -999, i].shape[0] / no_rows
        if(freq !=0):
            print("Column {}, freq {}".format(i, freq))
            ind_delete.append(i)
            
    X = np.delete(X, ind_delete, axis=1)
    return X


