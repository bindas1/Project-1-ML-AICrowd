# -*- coding: utf-8 -*-
"""File to obtain the best results that we obtained on AI Crowd."""

from implementations import *
from proj1_helpers import *
from split_expand_data import *
from clean_data_4_categories import *


DATA_TRAIN_PATH = '../data/train.csv' 
DATA_TEST_PATH = '../data/test.csv'
OUTPUT_PATH = '../data/submission.csv'

y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

#Divide the data in 4 categories depending on the value of PRI_jet_num (column 22), and remove all columns where standard variation is 0
tX1_ind = np.array(tX[:, 22]==0.)
tX1 = tX[tX[:, 22]==0.]
tX1 = clean_data(tX1)
y1 = y[tX[:, 22]==0.]

tX2_ind = np.array(tX[:, 22]==1.)
tX2 = tX[tX[:, 22]==1.]
tX2 = clean_data(tX2)
y2 = y[tX[:, 22]==1.]

tX3_ind = np.array(tX[:, 22]==2.)
tX3 = tX[tX[:, 22]==2.]
tX3 = clean_data(tX3)
y3 = y[tX[:, 22]==2.]

tX4_ind = np.array(tX[:, 22]==3.)
tX4 = tX[tX[:, 22]==3.]
tX4 = clean_data(tX4)
y4 = y[tX[:, 22]==3.]

# split data into train and test sets
X1_train, y1_train, X1_test, y1_test = split_data(tX1, y1, split_ratio=0.8)
X2_train, y2_train, X2_test, y2_test = split_data(tX2, y2, split_ratio=0.8)
X3_train, y3_train, X3_test, y3_test = split_data(tX3, y3, split_ratio=0.8)
X4_train, y4_train, X4_test, y4_test = split_data(tX4, y4, split_ratio=0.8)

# store d to later use with real test set
d=7

# create expanded X_train and X_test and normalize

#Category 1
X1_train_poly, X1_mu_train_poly, X1_std_train_poly = expand_and_normalize_X(X1_train, d)
X1_test_poly  = expand_X_cross_1_trigo(X1_test, d, 2)
X1_test_poly[:,1:] = (X1_test_poly[:,1:] - X1_mu_train_poly) / X1_std_train_poly

#Category 2
X2_train_poly, X2_mu_train_poly, X2_std_train_poly = expand_and_normalize_X(X2_train, d)
X2_test_poly  = expand_X_cross_1_trigo(X2_test, d, 2)
X2_test_poly[:,1:] = (X2_test_poly[:,1:] - X2_mu_train_poly) / X2_std_train_poly

#Category 3
X3_train_poly, X3_mu_train_poly, X3_std_train_poly = expand_and_normalize_X(X3_train, d)
X3_test_poly  = expand_X_cross_1_trigo(X3_test, d, 2)
X3_test_poly[:,1:] = (X3_test_poly[:,1:] - X3_mu_train_poly) / X3_std_train_poly

#Category 4
X4_train_poly, X4_mu_train_poly, X4_std_train_poly = expand_and_normalize_X(X4_train, d)
X4_test_poly  = expand_X_cross_1_trigo(X4_test, d, 2)
X4_test_poly[:,1:] = (X4_test_poly[:,1:] - X4_mu_train_poly) / X4_std_train_poly

# least squares for 4 categories
w1, loss_analytical1 = least_squares(y1_train, X1_train_poly)
w2, loss_analytical2 = least_squares(y2_train, X2_train_poly)
w3, loss_analytical3 = least_squares(y3_train, X3_train_poly)
w4, loss_analytical4 = least_squares(y4_train, X4_train_poly)

# predictions
p1 = predict_labels(w1, X1_test_poly)
p2 = predict_labels(w2, X2_test_poly)
p3 = predict_labels(w3, X3_test_poly)
p4 = predict_labels(w4, X4_test_poly)

accuracy1 = (np.mean(p1 == y1_test) * 100)
accuracy2 = (np.mean(p2 == y2_test) * 100)
accuracy3 = (np.mean(p3 == y3_test) * 100)
accuracy4 = (np.mean(p4 == y4_test) * 100)
# print the total accuracy
print("Accuracy on our test set for all 4 categories {}".format(np.sum([len(y1_test)*accuracy1, len(y2_test)*accuracy2, len(y3_test)*accuracy3, len(y4_test)*accuracy4])/(len(y1_test)+len(y2_test)+len(y3_test)+len(y4_test))))

# Obtaining results

_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX1_test_ind = np.array(tX_test[:, 22]==0.)
tX1_test = tX_test[tX_test[:, 22]==0.]
tX1_test = clean_data(tX1_test)

tX2_test_ind = np.array(tX_test[:, 22]==1.)
tX2_test = tX_test[tX_test[:, 22]==1.]
tX2_test = clean_data(tX2_test)

tX3_test_ind = np.array(tX_test[:, 22]==2.)
tX3_test = tX_test[tX_test[:, 22]==2.]
tX3_test = clean_data(tX3_test)

tX4_test_ind = np.array(tX_test[:, 22]==3.)
tX4_test = tX_test[tX_test[:, 22]==3.]
tX4_test = clean_data(tX4_test)


#Category 1
tX1_test_poly  = expand_X_cross_1_trigo(tX1_test, d, 2)
tX1_test_poly[:,1:] = (tX1_test_poly[:,1:] - X1_mu_train_poly) / X1_std_train_poly

#Category 2
tX2_test_poly  = expand_X_cross_1_trigo(tX2_test, d, 2)
tX2_test_poly[:,1:] = (tX2_test_poly[:,1:] - X2_mu_train_poly) / X2_std_train_poly

#Category 3
tX3_test_poly  = expand_X_cross_1_trigo(tX3_test, d, 2)
tX3_test_poly[:,1:] = (tX3_test_poly[:,1:] - X3_mu_train_poly) / X3_std_train_poly

#Category 4
tX4_test_poly  = expand_X_cross_1_trigo(tX4_test, d, 2)
tX4_test_poly[:,1:] = (tX4_test_poly[:,1:] - X4_mu_train_poly) / X4_std_train_poly

y_pred = np.empty(tX_test.shape[0])
y1_pred = predict_labels(w1, tX1_test_poly)
y2_pred = predict_labels(w2, tX2_test_poly)
y3_pred = predict_labels(w3, tX3_test_poly)
y4_pred = predict_labels(w4, tX4_test_poly)

y_pred[tX1_test_ind] = y1_pred
y_pred[tX2_test_ind] = y2_pred
y_pred[tX3_test_ind] = y3_pred
y_pred[tX4_test_ind] = y4_pred

# creating submissions for the test set
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

