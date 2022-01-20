#!/usr/bin/env python3

'''
Classification algorithm using scikit-learn's DecisionTreeClassifier and make confusion matrix
'''

import sys
import argparse
import pathlib
import math
from statistics import mode

import confusion

import pandas as pd
import numpy as np

import random

from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns



## get values from command line if given
argparser = argparse.ArgumentParser(
        description="Predict class with DecisionTreeClassifier; make confusion matrix")
argparser.add_argument("-c", "--classes", action="store", default=None,
		help="A comma-separated list of class names to report in "
			"evalulation metrics.  If not supplied, class names "
			"based on integer numbers starting at zero will be used.")
argparser.add_argument("dirname",
        type=pathlib.Path,
        help="Sets the data directory -- alterate to -I")

args = argparser.parse_args(sys.argv[1:])


if args.classes is None:
    CLASS_NAMES = None
else:
    CLASS_NAMES = args.classes.split(',')


DATA_DIRNAME=args.dirname


print(" . Loading....")
X_data = pd.read_csv("%s/X_split.csv" % DATA_DIRNAME)
y_data = pd.read_csv("%s/y_split.csv" % DATA_DIRNAME)

# get column names for feature labels
feature_list = X_data.columns

## need to convert X to numpy arrays and y to a column vector
X_data = X_data.to_numpy()

# convert from 1-d array to column vector
y_data = np.ravel(y_data.to_numpy())


##
## Use "LeaveOneOut" to find best first split for our DecisionTree
##

print(" . Creating DecisionTree model")

#Set cross-validation procedure
loocv = LeaveOneOut()

# enumerate splits
y_true, y_pred = list(), list()

parameter_write = open('%s/tree_node0.csv' % DATA_DIRNAME, "w")
parameter_write.write("0Node_Feature,0Node_Threshold\n")

for train_index, test_index in loocv.split(X_data):
        #split data
        X_train, X_test = X_data[train_index, :], X_data[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]
        #fit model
        model = DecisionTreeClassifier(max_depth=1, class_weight={0: 1, 1: 100})
        model.fit(X_train, y_train)
        #evaluate model
        yhat = model.predict(X_test)
        #store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
        #save node0
        # Print best parameters for reference
        n_nodes = model.tree_.node_count
        feature = model.tree_.feature
        threshold = model.tree_.threshold
        parameter_write.write(str(feature[0]) + "," + str(threshold[0]) + "\n")


parameter_write.close()

##
## Save best first split
##

print("Saving tree split")
# Read in data
node_0 = pd.read_csv("%s/tree_node0.csv" % DATA_DIRNAME)
node_0np = node_0.to_numpy()

# Check feature & threshold
#unique_features = np.unique(node_0np[:,0])
split_1 = mode(node_0np[:,0])
sub_frame = node_0.loc[node_0['0Node_Feature'] == split_1,]
sub_frame = sub_frame.to_numpy()
thresh_1 = sum(sub_frame[:,1])/len(sub_frame[:,1])
percent_split = len(sub_frame[:,0])*100/len(node_0np[:,0])
print("%.2f of LOOCV trials split as follows:" % percent_split)

print("Feature: " + str(split_1) + "; Mean threshold: " + str(thresh_1))


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

outcome.to_csv('%s/tree_outcome0.csv' % DATA_DIRNAME, index=False)



# Calculate confusion matrix and print it
cm = metrics.confusion_matrix(y_true, y_pred)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data,
		filename="%s/tree_confusion0.txt" % DATA_DIRNAME)


##
## Save model of all samples
##

model_all = DecisionTreeClassifier(max_depth=1, class_weight={0: 1, 1: 100})
model_all.fit(X_data, y_data)

tree.plot_tree(model_all,class_names=CLASS_NAMES, filled=True, fontsize=7, label='none', impurity=False, feature_names=feature_list)
plt.show()