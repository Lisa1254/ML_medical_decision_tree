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

from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
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
argparser.add_argument("-n", "--nodes", action="store", type=int, default=3,
        help="Node depth of trees")
argparser.add_argument("-k", "--features", action="store", type=int, default=3,
        help="Number of features to keep in selection")
argparser.add_argument("dirname",
        type=pathlib.Path,
        help="Sets the data directory -- alterate to -I")

args = argparser.parse_args(sys.argv[1:])


if args.classes is None:
    CLASS_NAMES = None
else:
    CLASS_NAMES = args.classes.split(',')


DATA_DIRNAME=args.dirname

k_feat=args.features
OUT_nodes=args.nodes
STR_nodes=str(OUT_nodes)

print(" . Loading....")
X_data = pd.read_csv("%s/X_split1.csv" % DATA_DIRNAME)
y_data = pd.read_csv("%s/y_split1.csv" % DATA_DIRNAME)

# get column names for feature labels
feature_list = X_data.columns

## need to convert X to numpy arrays and y to a column vector
X_data = X_data.to_numpy()

# convert from 1-d array to column vector
y_data = np.ravel(y_data.to_numpy())

##
## Select k best features
##
ch2 = SelectKBest(chi2, k=k_feat)
X_new = ch2.fit_transform(X_data, y_data)
best_feat_ind = ch2.get_support(indices=True)
best_feat = [feature_list[i] for i in best_feat_ind]
print("k best features: " + str(best_feat))


print(" . Creating DecisionTree model")

#Set cross-validation procedure
loocv = LeaveOneOut()

# enumerate splits
y_true, y_pred = list(), list()

all_trees_write = open('%s/%snode_all_trees.txt' % (DATA_DIRNAME, STR_nodes), "w")


for train_index, test_index in loocv.split(X_new):
        #split data
        X_train, X_test = X_new[train_index, :], X_new[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]
        #fit model
        model = DecisionTreeClassifier(max_depth=OUT_nodes)
        model.fit(X_train, y_train)
        #evaluate model
        yhat = model.predict(X_test)
        #store
        y_true.append(y_test[0])
        y_pred.append(yhat[0])
        # Print tree for reference
        n_nodes = model.tree_.node_count
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right
        feature = model.tree_.feature
        threshold = model.tree_.threshold
        
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]
        while len(stack) > 0:
            node_id, depth = stack.pop()
            node_depth[node_id] = depth

            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True
        all_trees_write.write("The binary tree structure has {n} nodes and has the following tree structure:\n".format(n=n_nodes))

        for i in range(n_nodes):
            if is_leaves[i]:
                all_trees_write.write("{space}node={node} is a leaf node.\n".format(space=node_depth[i] * "\t", node=i))
            else:
                all_trees_write.write("{space}node={node} is a split node: "
                    "go to node {left} if X[:, {feature}] <= {threshold} "
                    "else to node {right}.\n".format(
                        space=node_depth[i] * "\t",
                        node=i,
                        left=children_left[i],
                        feature=feature[i],
                        threshold=threshold[i],
                        right=children_right[i]))




all_trees_write.close()


##
## Evaluate the results
##
print(" . Performing evaluation from LOOCV:")


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

outcome.to_csv('%s/%snode_tree_outcome1loocv.csv' % (DATA_DIRNAME, STR_nodes), index=False)



# Calculate confusion matrix and print it
cm = metrics.confusion_matrix(y_true, y_pred)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data,
		filename="%s/%snode_tree_confusion1loocv.txt" % (DATA_DIRNAME, STR_nodes))



##
## Save model of all samples
##

model_all = DecisionTreeClassifier(max_depth=OUT_nodes)
model_all.fit(X_new, y_data)

y_pred2 = model_all.predict(X_new)

##
## Evaluate the results
##
print(" . Performing evaluation from Complete Tree:")


# If these labels are correct, they will match the ones in y_test
prediction_accuracy = metrics.accuracy_score(y_data, y_pred2)

unique_labels = np.unique(y_data)
if len(unique_labels) <= 2:
	prediction_precision = metrics.precision_score(y_data, y_pred2)
	prediction_recall = metrics.recall_score(y_data, y_pred2)
else:
	prediction_precision = metrics.precision_score(y_data, y_pred2,
			average="macro")
	prediction_recall = metrics.recall_score(y_data, y_pred2,
			average="macro")

print('Performance metrics:  accuracy %.2f, precision %.2f, recall %.2f'
        % (prediction_accuracy, prediction_precision, prediction_recall))


outcome = pd.DataFrame()
outcome['predicted'] = y_pred2
outcome['actual'] = y_data

outcome.to_csv('%s/%snode_tree_outcome1all.csv' % (DATA_DIRNAME, STR_nodes), index=False)



# Calculate confusion matrix and print it
cm = metrics.confusion_matrix(y_true, y_pred)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data)
confusion.print_confusion_matrix(cm, CLASS_NAMES, y_data,
		filename="%s/%snode_tree_confusion1all.txt" % (DATA_DIRNAME, STR_nodes))


#Display tree
tree.plot_tree(model_all,class_names=CLASS_NAMES, filled=True, fontsize=7, label='none', impurity=False, feature_names=best_feat)
plt.show()