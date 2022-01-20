#!/usr/bin/env python3

'''
Convert the UCI data from parent directory.
Mapping of classification variable will be binary.
Here it was easier simply to encode all the names for the columns
in this file rather than do complicated parsing from the 
"hcvdat0.names" file.
'''


import pandas as pd
import numpy as np

OUTPUT_FILENAME = "Source_Data/hep_c_binary.csv"
MAPPING_FILENAME = "Source_Data/hep_c_binary_mappings.csv"

DESIRED_COLUMN_NAMES = [
		"ID",
		"Category",
		"Age",
		"Sex",
		"ALB",
		"ALP",
		"ALT",
		"AST",
		"BIL",
        "CHE",
        "CHOL",
        "CREA",
        "GGT",
        "PROT" ]

# Category is represented with the following strings:
## 0=Blood Donor
## 0s=suspect Blood Donor
## 1=Hepatitis
## 2=Fibrosis
## 3=Cirrhosis
# For binary mapping, 0 & 0s Blood Donors will be mapped to 0, and 1,2,3 Disease State will be mapped to 1

print("Loading CSV files....")
data_set = pd.read_csv("Source_Data/hcvdat0.data", names=DESIRED_COLUMN_NAMES, header=0)

print("CSV files loaded")

# calculate new values for all three "Sex" values
mapping_conditions = [
			(data_set['Category'] == "0=Blood Donor"),
			(data_set['Category'] == "0s=suspect Blood Donor"),
            (data_set['Category'] == "1=Hepatitis"),
            (data_set['Category'] == "2=Fibrosis"),
            (data_set['Category'] == "3=Cirrhosis"),
		]
mapping_values = [ 0, 0, 1, 1, 1 ]
data_set['Category'] = np.select(mapping_conditions, mapping_values)

label_mappings = pd.DataFrame( { "Label": [ "0=Blood Donor", "0s=suspect Blood Donor", 
                                    "1=Hepatitis", "2=Fibrosis", "3=Cirrhosis" ],
			"MappedTo": [ "0", "0", "1", "1", "1" ] } )


# Don't need column of sample ids, age, or sex
data_set_drop = data_set.drop(columns=['ID', 'Age', 'Sex'])


print("Head of data set")
print(data_set_drop.head())

data_set_drop.to_csv(OUTPUT_FILENAME, index=False)
print(f"Wrote {OUTPUT_FILENAME} file with labelled data")

label_mappings.to_csv(MAPPING_FILENAME, index=False)
print(f"Wrote {MAPPING_FILENAME} file with labelled data")