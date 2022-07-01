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
from src.utils.helper_data_loaders import load_tsv
from sklearn.metrics import classification_report
from src.utils.helper_merge_dfs import merge_data
from src.utils.Model_Evaluation_TOOLS import *


# function, ........................................
def train_and_evaluate_models(
    model_name,
    dataset_name_list, 
    rand_nr_list, 
    param_grid, 
    model, 
    path_in, 
    path_out,
    none_at=None,
    verbose=False,
    b_verbose=True,
    tr=0.5,
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
        . verbose; bool, if true provides info and roc curve plot on every model
        . b_verbose; bool, if true, provides basic info on training process
        . tr=0.5; threshold used to classify the results, from predict_proba vector, 
        
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
    model_statistics = [] # roc auc, acc, spec, and sensitivity, later turned into pd.dataframe

    model_id=-1 # to iniciate
    for dataset_name in dataset_name_list:
        for rand_nr in rand_nr_list:
            if b_verbose==True:
                print("\n\nTrainig:", dataset_name, " V",rand_nr, "\nmodel_id: ", end="")
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
                
                # load data, ................................................
                ''' here I am dealing with small data, 
                    otherwise I woudl load them only once, per dataset/rand_nr
                    and check for modificaitons, 
                ''' 
                x_train, x_valid, x_test, y_train, y_valid = load_dataset_dct(
                    dataset_name, path_in, rand_nr, none_at=none_at)

                # fit model and create predictions , .......................

                # . baseline - most frequent baseline is build in the model evaluation function, 

                # . set params & fit model
                model.set_params(**model_params)
                model.fit(x_train, y_train)

                # . generate and store predictions
                model_predictions.append({
                    **model_fingerprint,
                    'predictions_train': prediction_table(model, x_train, y_train, **model_fingerprint),
                    'predictions_valid': prediction_table(model, x_valid, y_valid, **model_fingerprint),
                    'predictions_test': prediction_table(model, x_test, None, **model_fingerprint),
                })

                # evaluate the model, .....................................          

                # . caluclate stats and make ROC plots
                one_model_stats = model_stats_and_plot_ROC_curves(
                    x = x_valid, 
                    y = y_valid, 
                    y_hat = model.predict(x_valid), 
                    y_hat_probs = model.predict_proba(x_valid)[:,1], 
                    model_fingerprint = model_fingerprint,
                    positive_class=1,
                    create_plot=verbose,
                    tr=tr
                )
                model_statistics.append(one_model_stats)

                if verbose==True:
                    display(pd.DataFrame([one_model_stats]))
                else:
                    pass


    # save the results, .....................................          

    # setup
    os.chdir(path_out)
    prefix = f'{model_name}'
    print("\n\nPWD: ", path_out)


    # . save predictions
    fname = f"{prefix}__model_predictions_list.p"
    with open(fname, 'wb') as file: # wb - write binary,
            pickle.dump(model_predictions, file) 
    print("SAVED: ", fname)


    # . save model params
    fname = f"{prefix}__model_parameters_df.p"
    data = pd.DataFrame(model_parameters)
    with open(fname, 'wb') as file: # wb - write binary,
            pickle.dump(data, file)           
    print("SAVED: ", fname)         


    # . save model stats evaluation
    fname = f"{prefix}__model_statistics_df.p"
    data = pd.DataFrame(model_statistics)
    with open(fname, 'wb') as file: # wb - write binary,
            pickle.dump(data, file)           
    print("SAVED: ", fname)   


    # return, .....................................          
    return pd.DataFrame(model_statistics), model_predictions, pd.DataFrame(model_parameters)
    

            
            
            
            
            