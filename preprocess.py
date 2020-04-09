# -*- coding: utf-8 -*-
'''
Created on Thu Apr  9 11:31:29 2020

@author: rriccilopes
'''

# Pre-process data
import os
import numpy as np
import scipy.signal as sps

# Folder with the data
base_dir = 'C:/Data/Age_and_sex/'
# Folder to save the processed data
output_dir = 'C:/Data/processed/'
label_list = ['male', 'female']

# Iterate over all data
for label in label_list:
    for file_path in os.listdir(base_dir + label):
        
        # Load ecg
        ecg = np.load(base_dir + label + '/' + file_path)
        
        # Check sampling != 2500. Resample
        if ecg.shape[1] != 2500:
            temp_lead = []
            for lead in ecg[:8]:
                res = sps.resample(lead, 2500)
                temp_lead.append(res)
            ecg = np.array(temp_lead)
        
        # Transpose and select first 8 leads
        ecg = ecg.T[:, :8]

        # Save ecg
        np.save(output_dir + label + '/' + file_path, ecg)

#%% Split data
import os
from shutil import copyfile
from sklearn.model_selection import train_test_split

# Base directory with data
## ATTENTION: THE BASEDIR HERE IS THE OUTPUTDIR FROM PREVIOUS STEP
## AND OUTPUT_DIR CAN BE ANYWHERE (I USED THE BASEDIR FROM PREVIOUS STEP)

# Folder with processed data
base_dir = 'C:/Data/processed/'
# Folder to save the splits
output_dir = 'C:/Data/Age_and_sex/'
label_list = ['male', 'female']


# TODO: Test it (written without running)
# Create train, test and validation folder
for i in ['train/', 'test/', 'validation/']:
    for label in label_list:
        if not os.path.exists(output_dir + i + label):
            os.makedirs(output_dir + i + label)

# List all the files and labels
all_file_list = []
all_label_list = []
for label in label_list:
    folder_files = os.listdir(base_dir + label)
    all_file_list = all_file_list + folder_files
    all_label_list = all_label_list + ([label] * len(folder_files))

del label, folder_files

# And split the data (train / test)
X_train, X_test, y_train, y_test = train_test_split(
        all_file_list, all_label_list, test_size=0.20,
        random_state=42, stratify=all_label_list)

# Split the data (train / validation)
X_train, X_val, y_train, y_val= train_test_split(
        X_train, y_train, test_size=0.20,
        random_state=42, stratify=y_train)

# Copy files from base_dir (folder with data processed) to output folders
for sample, label in zip(X_train, y_train):
    copyfile(base_dir + label + '/' + sample, output_dir + 'train/' + label + '/' + sample)
del sample, label    

for sample, label in zip(X_test, y_test):
    copyfile(base_dir + label + '/' + sample, output_dir + 'test/' + label + '/' + sample)
del sample, label
    
for sample, label in zip(X_val, y_val):
    copyfile(base_dir + label + '/' + sample, output_dir + 'validation/' + label + '/' + sample)
del sample, label