#!/bin/bash

#
# Runs an analysis pipeline for DecisionTreeClassifier under multiple conditions
#

### Safety standards
# -u : tell bash to fail if any variable is not set
# -e : tell bash to fail if any command fails (unless in an if)
# -o pipefail : tell bash to fail if part of a pipe fails (needs -e)
set -e
set -u
set -o pipefail

# save the directory that our script it in so that we can find
# the tools
SCRIPTDIR=`dirname $0`

# -x Turn tracing on to make clearer what is happening
set -x 

##
## Search for first best split for donor/disease to determine if further tests are necessary
## Assumes "Source_Data/hep_c_binary.narm.csv" already created through runpipe_svc.sh
##

## If running before runpipe_svc.sh, uncomment this to prep the data
## Prep data for binary
#python3 ${SCRIPTDIR}/convert_data_binary.py
#grep -v ",," Source_Data/hep_c_binary.csv > Source_Data/hep_c_binary.narm.csv

# Split data to X and y
python3 ${SCRIPTDIR}/scale_pca_split.py \
        --tag "Tree" -l "Category" \
        "Source_Data/hep_c_binary.narm.csv"

# Determine tree split1 from DT
python3 ${SCRIPTDIR}/tree_split0.py \
        -c "Donor,Disease" "Tree-output"

##
## Finish spliting tree for disease severityt
## Assumes "Source_Data/hep_c_multiclass.narm.csv" already created through runpipe_svc.sh
##

## If running before runpipe_svc.sh, uncomment this to prep the data
## Prep data for multiclass
#python3 ${SCRIPTDIR}/convert_data_multiclass.py
#grep -v ",," Source_Data/hep_c_multiclass.csv > Source_Data/hep_c_multiclass.narm.csv

# Split data to X and y with given threshold
#If no threshold supplied, it will be calculated from the tree_node0.csv file in prev step
python3 ${SCRIPTDIR}/split_by_thresh.py \
        --tag "Tree" -l "Category" \
        "Source_Data/hep_c_multiclass.narm.csv"

# Determine subsequent tree splits from DT with node depth=3
python3 ${SCRIPTDIR}/tree_split1.py \
        -k 4 -n 3 "Tree-output" \
        -c "Donor,Hepatitis,Fibrosis,Cirrhosis"

# Determine subsequent tree splits from DT with node depth=2
python3 ${SCRIPTDIR}/tree_split1.py \
        -k 4 -n 2 "Tree-output" \
        -c "Donor,Hepatitis,Fibrosis,Cirrhosis"