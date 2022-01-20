#!/usr/bin/env python3

'''
Classification algorithm using scikit-learn's SVC and make confusion matrix
'''

import sys
import argparse
import pathlib
import math

import confusion

import pandas as pd
import numpy as np

import random

from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns



## get values from command line if given
argparser = argparse.ArgumentParser(
        description="Predict class with SVC; make confusion matrix")
argparser.add_argument("-c", "--classes", action="store", default=None,
		help="A comma-separated list of class names to report in "
			"evalulation metrics.  If not supplied, class names "
			"based on integer numbers starting at zero will be used.")
argparser.add_argument("-f", "--fig", action="store_true",
        help="Stores the confusion matrix in a PDF figure")
argparser.add_argument("dirname",
        type=pathlib.Path,
        help="Sets the data directory -- alterate to -I")

args = argparser.parse_args(sys.argv[1:])


if args.classes is None:
    CLASS_NAMES = None
else:
    CLASS_NAMES = args.classes.split(',')


MAKE_FIG=args.fig
DATA_DIRNAME=args.dirname



print(" . Loading....")
X_data = pd.read_csv("%s/X_split.csv" % DATA_DIRNAME)
y_data = pd.read_csv("%s/y_split.csv" % DATA_DIRNAME)

## need to convert X to numpy arrays and y to a column vector
X_data = X_data.to_numpy()

# convert from 1-d array to column vector
y_data = np.ravel(y_data.to_numpy())


##
## Use "LeaveOneOut" to find good values for our SVC
##

print(" . Creating SVC model")

#Set cross-validation procedure
loocv = LeaveOneOut()

# enumerate splits
y_true, y_pred = list(), list()

for train_index, test_index in loocv.split(X_data):
        #split data
        X_train, X_test = X_data[train_index, :], X_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]
        #fit model
        model = SVC()
        model.fit(X_train, y_train)
        #evaluate model
        yhat = model.predict(X_test)
        #store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])



##
## Evaluate the results
##
print(" . Performing evaluation:")


# If these labels are correct, they will match the ones in y_test
prediction_accuracy = metrics.accuracy_score(y_true, y_pred)

unique_labels = np.unique(y_data)
if len(unique_labels) <= 2:
	prediction_precision = metrics.precision_score(y_true, y_pred)
	prediction_recall = metrics.recall_score(y_true, y_pred)
else:
	prediction_precision = metrics.precision_score(y_true, y_pred,
			average="macro")
	prediction_recall = metrics.recall_score(y_true, y_pred,
			average="macro")

print('Performance metrics:  accuracy %.2f, precision %.2f, recall %.2f'
        % (prediction_accuracy, prediction_precision, prediction_recall))


outcome = pd.DataFrame()
outcome['predicted'] = y_pred
outcome['actual'] = y_true

outcome.to_csv('%s/svc_outcome.csv' % DATA_DIRNAME, index=False)


# Calculate confusion matrix and print it
cm = metrics.confusion_matrix(y_true, y_pred)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data,
		filename="%s/svc_confusion.txt" % DATA_DIRNAME)


if MAKE_FIG:
	confusion.confusion_matrix_heatmap(
            '%s/svc_confusion_heatmap.pdf' % DATA_DIRNAME,
            cm, CLASS_NAMES, y_data)
        
