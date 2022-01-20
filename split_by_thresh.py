#!/usr/bin/env python3

'''
Split data into X and y, with options to 
Standardize the data and apply PCA if requested
'''

import sys
import os
import argparse
import pathlib
import math
from statistics import mode

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

###
### Set up environment
###

argparser = argparse.ArgumentParser(
        description="Separate X and y, with options for to exclude by an attribute threshold.")
argparser.add_argument("--tag", action="store",
		default=None, type=str,
		help="Give an additional tag used in output directory name "
			"creation.")
argparser.add_argument("-l", "--label", action="store",
        default=None, type=str,
        help="Indicate the label column -- if not given, the last "
			"column is assumed to be the label")
argparser.add_argument("-f", "--feature", action="store", type=int,
        help="Feature of first split.")
argparser.add_argument("-t", "--threshold", action="store", type=float,
        help="Threshold of first split.")
argparser.add_argument("-L", "--lower", action="store_true",
        help="If selected will indicate that samples of interest have "
            "lower values than threshold. Default is greater than.")
argparser.add_argument("filename", action="store",
		help="The filename to use for input, which is assumed to contain "
			"all available data.")

args = argparser.parse_args(sys.argv[1:])

if args.label is None:
	LABEL=None
else:
	LABEL=args.label

if args.tag is None:
	TAG = ""
else:
	TAG = "%s-" % args.tag


DATAFILENAME=args.filename

data_dirname = "%soutput" % TAG
if not os.path.exists(data_dirname):
    os.mkdir(data_dirname)



LESS_THAN = args.lower

node_0 = pd.read_csv("%s/tree_node0.csv" % data_dirname)
node_0np = node_0.to_numpy()
split_1 = mode(node_0np[:,0])
sub_frame = node_0.loc[node_0['0Node_Feature'] == split_1,]
sub_frame = sub_frame.to_numpy()
thresh_1 = sum(sub_frame[:,1])/len(sub_frame[:,1])

if args.feature is None:
	split_feature=int(split_1)
else:
	split_feature=args.feature

if args.threshold is None:
	split_thresh = thresh_1
else:
	split_thresh = args.threshold

###
### Import data
###

print(f"Loading '{DATAFILENAME}'")
labelled_data = pd.read_csv(DATAFILENAME)
print("Data loaded")

if LABEL is None:
	LABEL = labelled_data.columns[-1]
	print(f" . Using '{LABEL}' as label")
elif LABEL not in labelled_data.columns:
	print(f"Error: Data from '{DATAFILENAME}' has no column named '{LABEL}'",
			file=sys.stderr)
	sys.exit(1)

split_name = labelled_data.columns[split_feature+1]

###
### Separate samples of interest
###

if LESS_THAN:
    keep_frame = labelled_data[labelled_data[split_name] <= split_thresh]
    out_frame = labelled_data[labelled_data[split_name] >= split_thresh]
else:
    keep_frame = labelled_data[labelled_data[split_name] >= split_thresh]
    out_frame = labelled_data[labelled_data[split_name] <= split_thresh]


out_frame.to_csv("%s/reserved_samples.csv" % data_dirname, index=False)

###
### Split X and y
###




# Split off the label column into a separate vector
#
# At this point X is a matrix of measures (feature values),
# and y is the vector of labels describing the rows in the X matrix

X_data, y_data = keep_frame.drop(columns=[LABEL]), keep_frame[LABEL]

X_data.to_csv("%s/X_split1.csv" % data_dirname, index=False)
y_data.to_csv("%s/y_split1.csv" % data_dirname, index=False)