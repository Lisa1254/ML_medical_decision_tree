#!/bin/bash

#
# Runs an analysis pipeline for SVC() classifier under multiple conditions
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

## Prep data for binary
python3 ${SCRIPTDIR}/convert_data_binary.py
grep -v ",," Source_Data/hep_c_binary.csv > Source_Data/hep_c_binary.narm.csv

### BINARY, SCALE ONLY
# split the data
python3 ${SCRIPTDIR}/scale_pca_split.py \
        --tag "Binary_scaled" -l "Category" -S \
        "Source_Data/hep_c_binary.narm.csv"

#SVC classifier
python3 ${SCRIPTDIR}/classify_svc.py \
        --fig -c "Donor,Disease" \
        "Binary_scaled-output"

### BINARY, PCA
# split the data
python3 ${SCRIPTDIR}/scale_pca_split.py \
        --tag "Binary_PCA" -l "Category" -S --PCA --fig \
        "Source_Data/hep_c_binary.narm.csv"

#SVC classifier
python3 ${SCRIPTDIR}/classify_svc.py \
        --fig -c "Donor,Disease" \
        "Binary_PCA-output"




## Prep data for multiclass
python3 ${SCRIPTDIR}/convert_data_multiclass.py
grep -v ",," Source_Data/hep_c_multiclass.csv > Source_Data/hep_c_multiclass.narm.csv

### MULTICLASS, SCALE ONLY
# split the data
python3 ${SCRIPTDIR}/scale_pca_split.py \
        --tag "Multi_scaled" -l "Category" -S \
        "Source_Data/hep_c_multiclass.narm.csv"

#SVC classifier
python3 ${SCRIPTDIR}/classify_svc.py \
        --fig -c "Donor,Hepatitis,Fibrosis,Cirrhosis" "Multi_scaled-output"

### MULTICLASS, PCA
# split the data
python3 ${SCRIPTDIR}/scale_pca_split.py \
        --tag "Multi_PCA" -l "Category" -S --PCA --fig \
        "Source_Data/hep_c_multiclass.narm.csv"

#SVC classifier
python3 ${SCRIPTDIR}/classify_svc.py \
        --fig -c "Donor,Hepatitis,Fibrosis,Cirrhosis" "Multi_PCA-output"