#!/usr/bin/env python3

'''
Split data into X and y, with options to 
Standardize the data and apply PCA if requested
'''

import sys
import os
import argparse
import pathlib

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
        description="Separate X and y, with options for standardization and PCA")
argparser.add_argument("--tag", action="store",
		default=None, type=str,
		help="Give an additional tag used in output directory name "
			"creation.")
argparser.add_argument("-l", "--label", action="store",
        default=None, type=str,
        help="Indicate the label column -- if not given, the last "
			"column is assumed to be the label")
argparser.add_argument("-S", "--scale", action="store_true",
        help="Standardize data. "
            "Required in order to do PCA.")
argparser.add_argument("-P", "--PCA", action="store_true",
        help="Perform PCA based dimensionality reduction. "
            "See also --threshold. Does nothing unless -S is also given.")
argparser.add_argument("-t", "--threshold", action="store",
        default=0.90, type=float,
        help="Set the threshold for PCA based dimensionality reduction. "
            "Does nothing unless --PCA is also given.")
argparser.add_argument("-f", "--fig", action="store_true",
        help="Store the results of PCA dimensionality reduction "
            "in a PDF figure")
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

DO_PCA = args.PCA
SUFFICIENT_VARIANCE_EXPLAINED = args.threshold
if SUFFICIENT_VARIANCE_EXPLAINED > 1.0 or SUFFICIENT_VARIANCE_EXPLAINED < 0:
    argparser.print_help()
    sys.exit(1)

MAKE_PCA_FIG = args.fig
SCALE_DATA = args.scale

DATAFILENAME=args.filename

data_dirname = "%soutput" % TAG
if not os.path.exists(data_dirname):
    os.mkdir(data_dirname)

###
### Functions used below
###
def calculate_PCA_projection( \
            X_projected,
            y_split,
            sufficient_variance_explained_threshold):

    print(" . Performing Dimensionality Reduction using PCA")

    ###
    ### Use PCA to decrease dimensionality as estimated also on the
    ### data.
    ###


    pca_model = PCA() 
    X_projected = pca_model.fit_transform(X_projected)


    ## Now walk through the components, determining how much of the total
    ## variance is explained as we add each.  The PCA routine presents them
    ## in decreasing order, so the first will explain the most variance,
    ## the second the next most, etc.

    # Calculate the total variance as the sum of the variances in each
    # dimension
    total = sum(pca_model.explained_variance_)


    # Iterate summing each new explained variance until we have decided
    # that we have "enough"
    k=0
    variance_explained=0.0
    current_variance=0.0

    while variance_explained < sufficient_variance_explained_threshold:
        current_variance += pca_model.explained_variance_[k]
        variance_explained = current_variance / total
        k=k+1

    k_PCA_sufficient = k

    print(" . PCA completed: %.2f of variance explained at %d components of %d total"
            % (sufficient_variance_explained_threshold,
                    k_PCA_sufficient, X_n_cols))


    # Using only k_PCA_sufficent components, re-fit the data
    # (producing the same fit) but now transforming to the lower
    # k_PCA_sufficient dimensional space
    pca = PCA(n_components = k_PCA_sufficient)
    X_projected = pca.fit_transform(X_projected)


    # Extract the amount of variance explained as a cumulative sum
    # series over the number of components
    cumulative_variance_explained = pca.explained_variance_ratio_.cumsum()



    ###
    ### Create some (relatively arbitrary) headers so that the
    ### the resulting data can be stored as a dataframe
    ###
    new_headers = [ 'PC_%0d' % i for i in range(k_PCA_sufficient) ]

    if MAKE_PCA_FIG:
        print(" . Generating figures....")
        save_pca_explanation_figure(k_PCA_sufficient, variance_explained,
                cumulative_variance_explained, data_dirname)
        save_pca_scatterplot_figure(X_projected, y_split,
                data_dirname)


    return (new_headers, X_projected)


def save_pca_explanation_figure(k_PCA_sufficient, \
        variance_explained, \
        cumulative_variance_explained, \
        data_dirname):
        

    '''
    Make a figure to explain PCA results
    '''

    # Make a bar plot -- limit it to only k_PCA_sufficient bars wide
    # rather than the total number of assay components so that it is
    # readable
    plt.bar(range(k_PCA_sufficient), cumulative_variance_explained)

    # Put on some nice labels and title
    plt.ylabel("Cumulative Explained Variance")
    plt.xlabel("Principal Components")
    plt.title("%.2f%% of variance (> 90%%) is explained by the first %d columns"
            % (variance_explained * 100.0, k_PCA_sufficient))

    # save it to a file
    fig_filename = "%s/PCA-variance-explained.pdf" % data_dirname
    plt.savefig(fig_filename, bbox_inches="tight")

    # If you want to see it on the screen, uncomment this -- not that
    # in that case this program will stop here until you close the
    # window
    #plt.show()

def save_pca_scatterplot_figure(X, y, data_dirname):
        
    '''
    Make a figure showing the PCA based scatterplot
    '''


    # y is a 1 x N matrix, and the columns of X are 1 x N vectors,
    # so we have to combine them in two steps
    plotdata = pd.DataFrame({"X_0" : X[:, 0], "X_1" : X[:, 1]})
    plotdata['label'] = y

    # Make a scatter plot of the first two axes of the X data,
    # with the colour (hue) based on the y values
    sns.relplot(x="X_0", y="X_1", hue="label", data=plotdata)

    # Put on some nice labels and title
    plt.xlabel("X_0 Components")
    plt.ylabel("X_1")
    plt.title("Scatter by first to principal components")

    # save it to a file
    fig_filename = "%s/PCA-data-scatter.pdf" % data_dirname
    plt.savefig(fig_filename, bbox_inches="tight")

    # If you want to see it on the screen, uncomment this -- not that
    # in that case this program will stop here until you close the
    # window
    #plt.show()

###
### Split X and y
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


# Split off the label column into a separate vector
#
# At this point X is a matrix of measures (feature values),
# and y is the vector of labels describing the rows in the X matrix
X_data, y_data = labelled_data.drop(columns=[LABEL]), labelled_data[LABEL]

# Since LOOCV will be used, no test/train splitting will be done

###
###  Perform scaling based on the training data to ensure
###  that some axes are not orders of magnitude bigger than
###  others.
###  Scaling data is not required for Decision Tree, and will be 
###  left out of that pipeline to make analysis easier.
###

if SCALE_DATA:
    print(" . Performing Standardized Scaling")
    _, X_n_cols = X_data.shape

    # perform a standard scaling on all X
    X_scaler = StandardScaler()
    X_projected = X_scaler.fit_transform(X_data)
    if DO_PCA:
        ## Calculate the PCA transform (below) to reduce dimensions
        (new_headers, X_projected) = \
                calculate_PCA_projection(
                        X_projected,
                        y_data,
                        SUFFICIENT_VARIANCE_EXPLAINED)
    else:
        #create headers that indicate we standardized
        new_headers = [ 'SCALED_%s' % header for header in X_data.columns ]
    # create dataframe from the raw array data created above
    df_X_split = pd.DataFrame(data=X_projected, columns=new_headers)
    ## Store the results
    df_X_split.to_csv("%s/X_split.csv" % data_dirname, index=False)
else:
    X_data.to_csv("%s/X_split.csv" % data_dirname, index=False)


## Store the results
y_data.to_csv("%s/y_split.csv" % data_dirname, index=False)