# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

from clean_data import *

DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../data/submission.csv'


def load_clean_data(data_path=DATA_TRAIN_PATH):
    """Loads the training data and cleans it
    Returns both y and improved tX with features without missing values.
    """
    y, tX, ids = load_csv_data(data_path)
    tX_clean, _ = update_X(tX, bound_delete=0.9, bound_change=0.05)
    return y, tX_clean

def obtain_results(weights, data_path=DATA_TEST_PATH, output_path=OUTPUT_PATH):
    _, tX_test, ids_test = load_csv_data(data_path)

    # clean outliers just like for training data
    tX_test = update_outliers(tX_test, [0, 4, 5, 6, 23, 26], [12, 24, 25, 27, 28])

    tX_test_poly  = expand_X(tX_test,d)
    tX_test_poly[:,1:]  = (tX_test_poly[:,1:]-mu_train_poly)/std_train_poly

    y_pred = predict_labels(weights, tX_test_poly)
    create_csv_submission(ids_test, y_pred, output_path)

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict_labels_log(weights, data):
    """Generates class predictions given weights, and a test data matrix
    !!! Remember that for this to work both y_train when training and y_test when testing should be set 0 and 1
    Instead of -1, 1 !!!
    """
    return np.where(datay@w > 0.5, 1, 0)

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
