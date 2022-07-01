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
#from src.utils.helper_data_loaders import load_tsv
from sklearn.metrics import classification_report
from src.utils.helper_merge_dfs import merge_data



# function, ...........................................................
def load_dataset_dct(dataset_name, path, rand_nr=0, none_at=None, verbose=False):
    '''helper function to allow trainign my models used to load train/testand validaiton data 
       stored in my custom data structures
       
       parameters
        . dataset_name; dataset name provided by the used eg: P17_G100
        . path; str, full path to input files
        . rand_nr, rand nr used to create test/valid data, used for full dataset name
        . none_at; None, or 1,2, for more see merge_data 
        . verbose; bool,
            
        returns:
        . x_train, x_valid, x_test, y_train, y_valid
    '''
    os.chdir(path)

    # . find names of the dataset_to_load 
    dataset_to_load = []
    for file in glob.glob(f"{dataset_name}_v{rand_nr}__transf_data_dct.p"):
        dataset_to_load.append(file)

    # . find name of the qc reports, 
    qc_report_to_load = []
    for file in glob.glob(f"{dataset_name}_v{rand_nr}__qc_reports_dct.p"):
        qc_report_to_load.append(file)

    # . load all of them 
    with open(dataset_to_load[0], 'rb') as file: 
            dataset_dct  = pickle.load(file)  
    with open(qc_report_to_load[0], 'rb') as file: 
            qc_report_dct  = pickle.load(file)  


    # extract & combine the data ..................................  
    if verbose==True:
        print(dataset_to_load)
    else:
        pass
    x_train = merge_data(dataset_dct["tpm_data"]["train"], dataset_dct["covariants_data"]["train"], none_at=none_at, verbose=verbose)
    y_train = dataset_dct['target_data']["train"]
    x_valid = merge_data(dataset_dct["tpm_data"]["test0"], dataset_dct["covariants_data"]["test0"], none_at=none_at, verbose=verbose)
    y_valid = dataset_dct['target_data']["test0"]
    x_test = merge_data(dataset_dct["tpm_data"]["test1"], dataset_dct["covariants_data"]["test1"], none_at=none_at, verbose=verbose)

    return x_train, x_valid, x_test, y_train, y_valid
    

    
# Function, ......................................................
def prediction_table(
    trained_model, x, y, model_name, model_id, cond_id, model_params, dataset_name, rand_nr
):
    "helper to create nice looking pd.series with prediciton results"
    # fill in y if it is missing
    if y is None:
        y = np.array([0]*x.shape[0])
    else:
        pass
    
    # predict class
    res_df = pd.DataFrame(
        np.c_[
            y,
            trained_model.predict(x),
            trained_model.predict_proba(x)[:, 1],
            # ...
            np.array([model_name]*y.shape[0]),
            np.array([model_id]*y.shape[0]),
            np.array([cond_id]*y.shape[0]),
            np.array([model_params]*y.shape[0]),
            np.array([dataset_name]*y.shape[0]),
            np.array([rand_nr]*y.shape[0]),
            
        ], columns=['y','yhat', 'yhat_prob', 
                    "model_name", "model_id", "cond_id", "model_params", "dataset_name", f"rand_nr"
                   ]
    )
    
    return res_df
             
 
# Function, ......................................................
def calculate_stats( trained_model, x, y, prefix, verbose=False):
    ''' helper to create nice looking pd.series with prediciton results
        its an anlternative to my other fucntion, that does the same, 
        but make much better plots and better organized summary, 
        and it does not require providing trained models, 
    '''

    # setup
    y=y
    y_hat=trained_model.predict(x)
    y_hat_prob=trained_model.predict_proba(x)[:, 1] # of class 1, p>0.5, its class=1

    # . count samples in eahc class
    unique, counts = np.unique(y, return_counts=True) 
    class_counts = dict(zip(unique, counts))
    
    
    # curves
    
    # .. calculate roc curve
    noskill_probs = [1 for _ in range(len(y))] # any value is ok, as long as it is the same for all
    noskill_fpr, noskill_tpr, _ = roc_curve(y, noskill_probs) # its no skill curce, in the middle, 
    model_fpr, model_tpr, _ = roc_curve(y, y_hat_prob)
        
    # .. calcluate pr curve    
    noskill_pr = len(y[y==1])/len(y)
    model_precision, model_recall, _ = precision_recall_curve(y, y_hat_prob)

    # store the results
    
    # .. basic stats
    stats = {
        f'acc_{prefix}' : model.score(x, y),
        f'f1_{prefix}' : f1_score(y, y_hat),
        f'recall_{prefix}' : recall_score(y, y_hat),
        f'precision_{prefix}' : precision_score(y, y_hat),
        f'ROC_AUC_{prefix}' : roc_auc_score(y, y_hat_prob),
        f'PRC_AUC_{prefix}':  auc(model_recall, model_precision),
        f'class_counts_{prefix}': class_counts
    }
    
    
    # make plots to see how does it looks
    if verbose==True:
        
        # ROC CURVE
        ns_probs = [1 for _ in range(len(y))] # any value is ok, as long as it is the same for all
        ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs) # its no skill curce, in the middle, 
        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(model_fpr, model_tpr, marker='.', label='Logistic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()  
        
        # PRC - precision-recall curves
        no_skill = len(y[y==1]) / len(y)
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        plt.plot(model_recall, model_precision, marker='.', label='Logistic')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.show()

    else:
        pass
        
    return stats


# Function, ......................................................
def model_stats_and_plot_ROC_curves(
    x, y, y_hat, y_hat_probs, model_fingerprint=dict(),
    plot_title="", figsize=(8,3), positive_class=1, negative_class=0,
    create_plot=False, tr=0.5
):
    """
        This function, train provided models
        subsequently, it plots ROC and Precision/Recal curves for these models
        and return a dictionary with statistics, such as ROc AUC and F1 for each of them
        ------------------
        . x, y;  imput data provided by create_data()
        . y_hat, y_hat_probs, prediction and predict proba from the models, 
        . plot_title_; str
        . figsize; tuple, eg (12,5) for plot size 
        . positive_class; int, 0, or 1, class name used as positive in confucion matrices
        . create_plot; bool, if true, plots ROC and PR curves, 
        . tr; a threshold used to show current point on the plots, based on recalculated conf. matrix
        
    """    
    
    # reevaluate y_hat if customr trheshodl is provided, .............
    if tr==0.5:
        y_hat = y_hat
    else:
        #print([positive_class for x in y_hat_probs if x>=tr else negative_class])
        y_hat = [positive_class if x>=tr else negative_class for x in y_hat_probs]
    plot_title=f"{plot_title}, Tr={tr}"
    
    
    # Data for ROC and PPR curves, ...................................
    
    
    # (a) data fpr ROC curve
        
    # . first generate a no skill prediction (majority class), 
    "any value will do as long as it is the same for all targets"
    ns_probs = [positive_class for _ in range(x.shape[0])]          
        
    # . then, calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)
    model_fpr, model_tpr, roc_thresholds = roc_curve(y, y_hat_probs)   
    
    # . find confusiona matrix provided by the model as default
    idx = np.max(np.arange(roc_thresholds.shape[0])[roc_thresholds>=tr])
    default_model_fpr = model_fpr[idx]
    default_model_tpr = model_tpr[idx]
    
    # (b) data for precision-recall curve
    
    # . generate data for model predicting only one class (positive class)
    " its == to frequency of the class labelled as positive, typically the most frequent"
    no_skill = len(y[y==positive_class])/len(y)         
    
    # . cacluate pr curve
    model_precision, model_recall, pr_thresholds = precision_recall_curve(y, y_hat_probs) 
    try:
        idx = np.min(np.arange(pr_thresholds.shape[0])[pr_thresholds>=tr])
        default_model_precision = model_precision[idx]
        default_model_recall = model_recall[idx]        
    except:
        pass

    
    # (c) baseline model
    y_unique, y_counts = np.unique(y, return_counts=True) 
    y_hat_baseline = np.full((len(y)), y_unique[y_counts==y_counts.max()][0])
    
    
    # Collect stats for the table, ...................................
            
    # . count samples in each class in y and y_hat
    y_unique, y_counts = np.unique(y, return_counts=True) 
    y_hat_unique, y_hat_counts = np.unique(y_hat, return_counts=True) 
    
    # . stats table
    stat_results = {
            **model_fingerprint,
            "Acc_baseline": accuracy_score(y, y_hat_baseline),
            "Acc": accuracy_score(y, y_hat),
            "ROC_AUC": roc_auc_score(y, y_hat_probs),
            "PRC_AUC":  auc(model_recall, model_precision),
            "Recall": recall_score(y, y_hat),
            "Presision": precision_score(y, y_hat), 
            "F1": f1_score(y, y_hat),
            "counts_y": dict(zip(y_unique, y_counts)),
            "counts_y_hat": dict(zip(y_hat_unique, y_hat_counts)),
            "tr":tr
        }
    
    # Plot, ............................................. 
    if create_plot==True:
    
        # create plot with 2 subplots for ROC and PR curves
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.suptitle(plot_title)

        # . roc curve
        axs[0].plot(model_fpr, model_tpr, marker='.', label=model_name)
        axs[0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        axs[0].scatter(default_model_fpr, default_model_tpr, s=1000, marker="o", color="red")
        axs[0].scatter(default_model_fpr, default_model_tpr, s=10, marker="o", color="red", label="def. therhold")
    
        # . precision-recall curves
        axs[1].plot(model_recall, model_precision, marker='.', label=model_name)
        axs[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        try:
            axs[1].scatter(default_model_recall, default_model_precision, s=1000, marker="o", color="red")
            axs[1].scatter(default_model_recall, default_model_precision, s=10, marker="o", color="red", label="def. therhold")
        except:
            pass
        
        # labels & legend      
        axs[0].set_ylim(0,1)
        axs[0].set_title("ROC curve")
        axs[0].set_xlabel('False Positive Rate  FP/(FP+TN)')
        axs[0].set_ylabel('True Positive Rate TP/(TP+FN)')
        axs[0].legend()      

        axs[1].set_ylim(0,1)
        axs[1].set_title("Precision-Recall curve")    
        axs[1].set_xlabel('Recall TP/(TP+FN)')
        axs[1].set_ylabel('Precision TP/(TP+FP)')
        axs[1].legend()     

        sns.despine()
        fig.tight_layout()
        plt.show();  

    return stat_results









