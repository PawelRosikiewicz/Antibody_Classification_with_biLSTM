# ********************************************************************************** #
#                                                                                    #                       
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz(a)gmail.com                                                #
#                                                                                    #
#   License: MIT License                                                             #
#   Copyright (C) 2022.06.04 Pawel Rosikiewicz                                       #
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

# my custom fucntions, 
# from src.utils.helper_data_loaders import load_tsv # not used - i am loading preprocessed data stored in dct, 
from sklearn.metrics import classification_report
from src.utils.helper_merge_dfs import merge_data
from src.utils.Model_Evaluation_TOOLS import *




# Function, ..................................................................
def subset_Xy(x, y, subset=None):
    ''' create smaller subset of the data to allow fast trianing
        and parameter selection, additionally it shuffles the data
        . X, y; numpy 2D and 1D array, repsectively,
        . subset; None, or float(0,1), if None, the data are just shuffled, in X,y in the same order
        
        Comment: is subset==None.then the data are just shuffled,  
    '''  
    # work on copy
    x, y = x.copy(), y.copy() 
    
    # setup, always shuffle while subsetting
    if subset is None:
        subset=1
    else:
        pass
        
    # find and shuffle idx of a samller arrays,
    subset_size = int(np.ceil(x.shape[0]*subset))
    
    # be sure there is no repeats (in case of error) 
    if subset_size>x.shape[0]: 
        subset_size>x.shape[0]
    
    # select idx for a new array
    idx_subset = np.random.choice(np.arange(x.shape[0]), subset_size, replace=False)       
    
    # subset X, y arrays
    x, y = x[idx_subset,:], y[idx_subset]
    
    return x, y
    
    
       
# Function, ..................................................................
def load_pickled_data(path, file_name, subset=None, subset_all=True, shuffle=True, verbose=False):  
    ''' Loads and unpack pickled data, preparted with data preprocessing pipeline, stored in dictioraries,
        . path; str, full path to data files,
        . file_name; str, ile name to load
        . subset; bool, if True, test and valid are also subset and shuffled, 
        . subset_all; bool, if True, test and validation is also subsetted, 
        . shuffle; bool, if True, it imposes shuffle on train data, 
        
        Comment: if you wish to just shuffle the train data, no subsetting, no shuffle in test and validation
        use: "subset=None, subset_all=False, shuffle=True"
    '''
    # . load the pickle
    os.chdir(path)
    with open(file_name, 'rb') as file: 
        data_prep = pickle.load(file)      

    # . extract valid predictions    
    x_train, y_train = data_prep["X"]["train"], data_prep["y"]["train"]
    x_valid, y_valid = data_prep["X"]["valid"], data_prep["y"]["valid"]
    x_test, y_test = data_prep["X"]["test"], data_prep["y"]["test"]

    # create smaller subset of the data to allow fast trianing and parameter selection  
    if subset is None and shuffle is False:
        pass
    if subset is None and shuffle==True:
        x_train, y_train = subset_Xy(x_train, y_train, subset=None) 
    else:
        x_train, y_train = subset_Xy(x_train, y_train, subset=subset) 
        # you may also subset test and validaiton data
        if subset_all==True:
            x_valid, y_valid = subset_Xy(x_valid, y_valid, subset=subset) 
            x_test, y_test = subset_Xy(x_test, y_test, subset=subset) 
        else:
            pass
    
    # info
    if verbose==True:
        print(f"Loading: {file_name}")
        print(f"train - X:{x_train.shape}, y:{y_train.shape}")
        print(f"valid - X:{x_valid.shape}, y:{y_valid.shape}")
        print(f"test - X:{x_test.shape}, y:{y_test.shape}")
    else:
        pass

    return x_train, y_train, x_valid, y_valid,  x_test, y_test
    
    

# function, ........................................
def train_and_evaluate_models(
    model_name,
    dataset_name_list,
    rand_nr_list, 
    param_grid, 
    model, 
    path_in, 
    path_out,
    dataset_full_name=False,
    none_at=None,
    verbose=False,
    b_verbose=True,
    tr=0.5,
    save_predictions=True,
    subset=None,
    subset_all=False,
    shuffle_train=True,
    positive_class=1
    ):
    '''
        function, that allows trainig, and evaluating large number 
        of skleanr models wiht any number of paramters stored in Parameter grid
        it stores, predicitons for train, test and valudation fdata, stats on the model, 
        and model parameters,
        
        parameters:
        . model_name; str, 
        . dataset_name_list; list with dataset names that will be loaded,  
        . rand_nr_list; list wiht int, for versiotns of datasets, 
        . param_grid; parameter grid for the model, created wiht ParameterGrid() function, 
        . model, instanciated skleanr model, 
        . path_in, path_out; str, full path to input datasets, and outputs
        . none_at=None, see load_dataset_dct
        . dataset_full_name; bool, if True, custom name is used to load the data, not the names provided with the pipeline
        . verbose; bool, if true provides info and roc curve plot on every model
        . b_verbose; bool, if true, provides basic info on training process
        . tr=0.5; threshold used to classify the results, from predict_proba vector, 
        . save_predictions; bool, if True, dcit., with predictions for each model is saved,
        . subset; bool, if True, test and valid are also subset and shuffled, 
          (by default is subsets all test, train and valid datasets)
        . subset_all; bool, if True, reduces the size of validaitons and test subsets, with subset parameter
        . shuffle_train; bool, if True, it imposes shuffle on train data,
        . positive_class; int, indicates which class labels should be used for ROC AUC analysis and displayed in QC report
          * comment: you may recalulate it for every class label having predicitons saved
        
        returns:
        - in this order model_statistics, model_predictions, model_parameters
        . model_parameters - pandas df, 
        . model_predictions - list wiht dictionary for each model, 
        . model_statistics - pandas df, 
        
    '''
    
    # to silence error on depreciated function, 
    warnings.filterwarnings("ignore") # because skleanr generates waringing if precision and f1 are 0.0
    if b_verbose==True:
        print("----------------------------------------------------")
        print("model_name:", model_name)
        print("cond nr:", len(param_grid))
        print("dataset nr:", len(dataset_name_list))
        print("cv nr:", len(rand_nr_list))
        print("----------------------------------------------------", end="")
    else:
        pass
   
    # prepare lists for the results
    model_parameters = []
    model_predictions = []
    model_statistics_valid = [] # roc auc, acc, spec, and sensitivity, later turned into pd.dataframe
    model_statistics_test = []
    model_id=-1 # to iniciate
    
    # iterate over datasets, and their versions prepared with different sample combinations, 
    for dataset_name in dataset_name_list:
        for rand_nr in rand_nr_list:
            if b_verbose==True:
                print("\n\nTrainig:", dataset_name, " v",rand_nr)
            else:
                pass
            # load data .................................................
            if dataset_full_name==False:
                file_name = f"Processed_{dataset_name}_onehotenc_v{rand_nr}_dct.p"
            else:
                file_name = f"{dataset_name}_dct.p"
            x_train, y_train, x_valid, y_valid, x_test, y_test = load_pickled_data(
                path_in, file_name, subset=subset, subset_all=subset_all, 
                shuffle=shuffle_train, verbose=b_verbose
            )
            if b_verbose==True:
                print("\nmodel_id: ", end="")
            else:
                pass
            
            # initiate cond_id to help yourself with groupby function, 
            cond_id=-1
            for model_params in param_grid:

                # setup, ................................................
                model_id+=1
                cond_id+=1
                model_fingerprint ={
                    "model_name":model_name,
                    "dataset_name":dataset_name,
                    "rand_nr": rand_nr,
                    "model_id":model_id,
                    "cond_id":cond_id,
                    "model_params":model_params
                }
                # store params used for that model (to rerun it if required)
                model_parameters.append({
                    **model_fingerprint,
                    **model_params
                    })
                if b_verbose==True:
                    print(model_id, end=", ")
                else:
                    pass
                
                # fit model and create predictions , .......................

                # . baseline - most frequent baseline is build in the model evaluation function, 

                # . set params & fit model
                model.set_params(**model_params)
                model.fit(x_train, y_train)

                # . generate and store predictions
                model_predictions.append({
                    **model_fingerprint,
                    #'predictions_train': prediction_table(model, x_train, y_train, **model_fingerprint),
                    'predictions_valid': prediction_table(model, x_valid, y_valid, **model_fingerprint),
                    'predictions_test': prediction_table(model, x_test, y_test, **model_fingerprint),
                })

                # evaluate the model, .....................................          

                # . caluclate stats and make ROC plots
                one_model_stats_valid = model_stats_and_plot_ROC_curves(
                    x = x_valid, 
                    y = y_valid, 
                    y_hat = model.predict(x_valid), 
                    y_hat_probs = model.predict_proba(x_valid)[:,1], 
                    model_fingerprint = model_fingerprint,
                    positive_class=positive_class,
                    create_plot=verbose,
                    tr=tr
                )
                # . and the same for test data, so we dont have to do it later, 
                one_model_stats_test = model_stats_and_plot_ROC_curves(
                    x = x_test, 
                    y = y_test, 
                    y_hat = model.predict(x_test), 
                    y_hat_probs = model.predict_proba(x_test)[:,1], 
                    model_fingerprint = model_fingerprint,
                    positive_class=positive_class,
                    create_plot=verbose,
                    tr=tr
                )
                
                model_statistics_valid.append(one_model_stats_valid)
                model_statistics_test.append(one_model_stats_test)

                if verbose==True:
                    display(pd.DataFrame([one_model_stats_valid]))
                else:
                    pass


    # save the results, .....................................          

    # setup
    os.chdir(path_out)
    prefix = f'{model_name}'
    print("\n\nPWD: ", path_out)

    # . save predictions
    if save_predictions==True:
        fname = f"{prefix}__model_predictions_list.p"
        with open(fname, 'wb') as file: # wb - write binary,
                pickle.dump(model_predictions, file) 
        print("SAVED: ", fname)
    else:
        pass
        
    # . save model params
    fname = f"{prefix}__model_parameters_df.p"
    data = pd.DataFrame(model_parameters)
    with open(fname, 'wb') as file: # wb - write binary,
            pickle.dump(data, file)           
    print("SAVED: ", fname)         

    # . save model stats evaluation
    fname = f"{prefix}__model_statistics_valid_df.p"
    data = pd.DataFrame(model_statistics_valid)
    with open(fname, 'wb') as file: # wb - write binary,
            pickle.dump(data, file)           
    print("SAVED: ", fname)  
        
    # . save model stats evaluation
    fname = f"{prefix}__model_statistics_test_df.p"
    data = pd.DataFrame(model_statistics_test)
    with open(fname, 'wb') as file: # wb - write binary,
            pickle.dump(data, file)           
    print("SAVED: ", fname)      

    # return, .....................................          
    return pd.DataFrame(model_statistics_valid), pd.DataFrame(model_statistics_test), pd.DataFrame(model_parameters), model_predictions
    

            
            
            
            