# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

BOUND_DELETE=0.5
BOUND_CHANGE=0.05

def clean_data(X, bound_delete=BOUND_DELETE, bound_change=BOUND_CHANGE):
    """Calculate how many of the values of the columns are equal to -999
    
        Returns
        -------
        ind_del: indices of columns to delete (more than 50% -999)
        ind_change: indices of columns where we need to keep process the data
    """
    ind_del = []
    ind_change = []
    
    # number of rows
    no_rows = X.shape[0]
    
    for i in range(X.shape[1]):
        freq = X[X[:, i]== -999, i].shape[0] / no_rows
        print("Row {}, freq {}".format(i, freq))
        if(freq > bound_delete):
            ind_del.append(i)
        elif(freq> bound_change):
            ind_change.append(i)
    return ind_del, ind_change

# helper function to visualize indices, where we would like to change the values
# use: boxplot_columns(tX, ind_change)
def boxplot_columns(X, columns):
    """Plot boxplots of the columns without -999 values"""
    
    for column in columns:
        fig1, ax1 = plt.subplots()
        ax1.set_title('Boxplot index'+str(column))
        ax1.boxplot(X[X[:, column]!=-999, column])

def update_outliers(X, columns_medians, columns_means):
    """
    Fixes values in the columns of X
    
    Parameters
    ---------
    X - array to be changed
    columns_medians / columns_means - indices of columns to be changed accordingly
    
    Returns
    -------
    Updated X with median values in place of -999 in the specified columns_medians
    and mean values in place of columns_means
    """
    for column in columns_medians:
        col_median = np.median(X[X[:, column]!=-999, column])
        X[X[:, column]==-999, column]= col_median
        
    for column in columns_means:
        col_mean = X[X[:, column]!=-999, column].mean()
        X[X[:, column]==-999, column]= col_mean
    
    return X


def update_X(X, bound_delete=BOUND_DELETE, bound_change=BOUND_CHANGE):
	"""Handles all of the preprocessing before training"""
	ind_delete, ind_change = clean_data(X, bound_delete, bound_change)

	# [0, 4, 5, 6, 23, 26], [12, 24, 25, 27, 28]

	X = update_outliers(X, [0, 4, 5, 6, 23, 26], [12, 24, 25, 27, 28])
	# X = np.delete(X, ind_delete, axis=1)
	return X, ind_delete




