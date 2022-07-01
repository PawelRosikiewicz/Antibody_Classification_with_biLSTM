# ********************************************************************************** #
#                                                                                    #                    
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz(a)gmail.com                                                #
#                                                                                    #
#   License: MIT License                                                             #
#   Copyright (C) 2022.06.25 Pawel Rosikiewicz                                       #
#                                                                                    #
# Permission is hereby granted, free of charge, to any person obtaining a copy       #
# of this software and associated documentation files (the "Software"), to deal      #
# in the Software without restriction, including without limitation the rights       #
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell          #
# copies of the Software, and to permit persons to whom the Software is              #
# furnished to do so, subject to the following conditions:                           #
#                                                                                    # 
# The above copyright notice and this permission notice shall be included in all     #
# copies or substantial portions of the Software.                                    #
#                                                                                    #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR         #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,           #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE        #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER             #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,      #
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE      #
# SOFTWARE.                                                                          #
#                                                                                    #
# ********************************************************************************** #



#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# classifiers used
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

# pipeline and model selection
from sklearn import set_config
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline, Pipeline # like pipeline function, but give step names automatically, 
from sklearn.decomposition import PCA

# feature transformations
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer # creates custom transfomers
from sklearn.compose import ColumnTransformer # allows using different transformers to different columns
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer # skleanr transformers
from sklearn.preprocessing import RobustScaler # creates custom transfomers

# stats
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# import my custom functions, 
from src.utils.data_loaders import load_aa_sequences # loads sequences and provides basic info on data
from src.utils.data_loaders import load_data_for_ml # more advanced loader, that provides data labels, test and train data
from src.utils.eda_helpers import aa_seq_len_hist
from src.utils.eda_helpers import aa_seq_qc # QC table on loaded qq-seq data
from src.utils.qc_helpers import unique_aa_counts_hist
from src.utils.data_preprocessing_tools import create_aa_matrix
from src.utils.data_preprocessing_tools import prepare_aa_data_for_baseline
from src.utils.data_preprocessing_tools import  calc_aa_perc_per_pos
from src.utils.data_preprocessing_tools import calc_aa_number_per_pos


  
# Function, .............................................................
def pickle_saver(
    X_train, X_valid, X_test, 
    y_train, y_valid, y_test,
    path=None, fname=None, verbose=False
):
    ''' saves pickle with dictionary containgin X, and y files, in sub-dictionaries, 
        each contains train, valid, and test subsets, 
    '''
    # setup
    if path is not None:
        os.chdir(path)
    else: 
        pass
    
    if fname is None:
        fname="data_for_ml.p"
    else:
        pass

    # store all data in one dict
    results_dict= {
                "X":{
                    "train":X_train,
                    "valid":X_valid,
                    "test":X_test,
                },
                "y":{
                    "train":y_train,
                    "valid":y_valid,
                    "test":y_test,
                }  
    }
    
    # save pickle - its fast, but dont use if you dont trust 
    with open(fname, 'wb') as file: # wb - write binary,
        pickle.dump(results_dict, file) 
            
    # info
    if verbose==True:
        print("SAVING:", fname)
        print("PWD: ", os.getcwd())
        print("\n")
    else:
        pass            



# Function, .....................................................................................................................
def vlen_data_preprocessing_pipeline(data, data_test, parameters, path=None, random_nr_list=[0], one_hot_encode=True, verbose=False):
    ''' Custom piepline for creating one-hot encoded, aa-seq data matrix, for sklearn ML models,
        the data, are divided into train/test and validation subsets, with features vectores and target variable stored separately, 
        the data are preformatted for sklearn models
        . data,
        . data_test, 
        . parameters, dict, with parameters used by the pipeline:
            . "train_size":<float>; propotion of samples used for train data
            . "prepare_aa_data_for_baseline":{'tr':<int>, 'min_nr':<int>}; for more see help in prepare_aa_data_for_baseline
        . random_nr_list; list wiht int, used for np.random.seed function
        . verbose; it True, extensive informaiton on the process is provided
        . path; str, full path to dir where the data will be stored,
    '''
    # Test input df,
    assert type(data) == pd.DataFrame, "Incorrect obj type: x shoudl be pd.DataFrame"
    assert type(data_test) == pd.DataFrame, "Incorrect obj type: x shoudl be pd.DataFrame"

    # work on copies
    data = data.copy()
    y_test = data_test.iloc[:,1]
    data_test = data_test.iloc[:,0].copy()


    # iterate over dataset varinats, and random numbers,  provided
    for dataset_name, params in parameters.items():
        for random_nr in random_nr_list:
            # info:
            if verbose==True:
                print("\n\n.....................................................................")
                print("dataset_name: ",dataset_name)
                print("rand_nr:", random_nr)
                print("params:", params)
                print(".....................................................................", end="\n")
            else:
                pass

            # Part 1. create test/validation dataset, ....................
            
            # . set seed
            np.random.seed(random_nr)

            # . create idx, for test/valid datasets
            idx_list = list(range(data.shape[0]))
            np.random.shuffle(idx_list)

            # decide on how many 
            n = int(np.ceil(len(idx_list)*params["train_size"]))
            valid_idx = idx_list[n:len(idx_list)]; 
            train_idx = idx_list[0:n]; 

            # .create train dataset 
            '''hand made methods - I had some issues with indexes'''
            data_train = data.iloc[train_idx,0]
            y_train = data.iloc[train_idx,1]
            data_train.reset_index(inplace=True, drop=True)
            y_train.reset_index(inplace=True, drop=True)

            # . create validation dataset for that seed
            data_valid = data.iloc[valid_idx,0]
            y_valid = data.iloc[valid_idx,1]
            data_valid.reset_index(inplace=True, drop=True)
            y_valid.reset_index(inplace=True, drop=True)

            # . info
            if verbose==True:
                print(f"\n- PART 1 - CREATE TRAIN AND VALIDAITON SUBSETS")
                aa_seq_qc(data_train, "train subset")
                aa_seq_qc(data_valid, "validation subset")
                aa_seq_qc(data_test, "test subset")
            else:
                pass

            
            # Part 2. cleand and encode train data, ....................
            '''NO ENCODING FOR VLEN DATA 
                - duplicates were already removed
                - I kept the pipeline to have the same order off all the samples, 
                - and the same parameters
            '''
            X_train = data_train
            X_valid = data_valid
            X_test = data_test

            #. info
            if verbose==True:
                print(f"\n- PART 2 - second test - it sgodul be only one column in each dataset\n")
                print(f"train data: X.shape = {X_train.shape}, y.shape = {y_train.shape}")
                print(f"valid data: X.shape = {X_valid.shape}, y.shape = {y_valid.shape}")
                print(f"test data: X.shape = {X_test.shape}, y.shape = {y_test.shape}")
            else:
                pass        
            
            # . save files, 
            fname = f"{dataset_name}_expanded_v{random_nr}_dct.p"
            pickle_saver(
                X_train.values, X_valid.values, X_test.values, 
                y_train.values, y_valid.values, y_test.values,
                path=path, fname=fname, verbose=verbose
            )
 
              
            return X_train.values, X_valid.values, X_test.values, y_train.values, y_valid.values, y_test.values