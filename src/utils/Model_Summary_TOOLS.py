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




# helper funciton, ..............................................................
def load_predictions_and_stats(path, model_name_list, results_type="valid", verbose=True):
    """ helper function, working only in my notebook, used to load the data for notebook 03
        params names, correspond wiht object names in the notebook
        . path; str full path
        . model_name_list,; list wiht model names used to save files, eg knn, logreg, 
        . results_type; str, "valid","train", or "test"
    """
    
    # setup
    os.chdir(path)

    # empty objects 
    model_predictions_valid = []
    model_predictions_test = []

    # (a) load model_statistics
    for i, model_name in enumerate(model_name_list):

        # .. create file name
        file_name = f'{model_name}__model_statistics_{results_type}_df.p'

        # .. load the pickle
        with open(file_name, 'rb') as file: 
            if i==0: 
                model_statistics_df  = pickle.load(file)      
            else: 
                model_statistics_df = pd.concat([model_statistics_df, pickle.load(file)])

        if verbose==True:print("loading_df", i, file_name, "shape_total:",model_statistics_df.shape)
        else: pass


    # (b) load model_predictions for test & validation data 
    for i, model_name in enumerate(model_name_list):

        # .. create file name
        file_name = f'{model_name}__model_predictions_list.p'

        # .. load the pickle
        with open(file_name, 'rb') as file: 
            all_predictions = pickle.load(file)      

        # .. extract valid predictions    
        for idx in range(len(all_predictions)):
            model_predictions_valid.append(all_predictions[idx]["predictions_valid"])
            model_predictions_test.append(all_predictions[idx]["predictions_test"])

        if verbose==True:print("loading_df", i, file_name, "shape_total:",len(model_predictions_valid))
        else: pass

    return model_statistics_df, model_predictions_valid, model_predictions_test




# helper function, ..............................................................
def modify_model_statistics_df(model_statistics_df):
    '''helper function to generate nice-llooking df, 
       with mean values from each model from cv
       . model_statistics_df, its model_statistics_df pandas df, 
         used in the project
    '''
    example_nr=0
    
    # add global id 
    '''used to find model examples'''
    model_statistics_df = pd.concat([
        model_statistics_df, 
        pd.Series(list(range(model_statistics_df.shape[0])))
    ], axis=1)
    model_statistics_df.columns.values[model_statistics_df.shape[1]-1]="ID"

    # . get means from cv for each model
    """# check group names and size, list(grp.groups); grp.size()"""
    grp = model_statistics_df.groupby(by=["model_name", "dataset_name", 'cond_id'])

    # . get cv-mean values for each modl type
    model_statistics_means=[]
    for name, group in grp:
        model_statistics_means.append({
            "model_name": group.loc[:,"model_name"].iloc[0],
            "dataset_name": group.loc[:,"dataset_name"].iloc[0], 
            "ID": group.loc[:,"ID"].iloc[example_nr], # idx form the first example
            
            # stats, 
            "ROC_AUC":np.round(group.loc[:,"ROC_AUC"].mean(),3),
            'Presision':np.round(group.loc[:,"Presision"].mean(),3),
            'Recall':np.round(group.loc[:,"Recall"].mean(),3),
            'F1':np.round(group.loc[:,"F1"].mean(),3),
            
            # class examples
            "tr": np.round(group.loc[:,"tr"].mean(),3),
            "counts_y": group.loc[:,"counts_y"].iloc[example_nr], # idx form the first example
            "counts_y_hat": group.loc[:,"counts_y_hat"].iloc[example_nr], # idx form the first example   
            
            # MODEL PARAMS
            'model_params':  group.loc[:,'model_params'].iloc[example_nr] # idx form the first example
        })

    model_statistics_means = pd.DataFrame(model_statistics_means)
    
    return model_statistics_means
  

    
# helper function, ..............................................................    
def create_group_top_model_list(
    model_statistics_means, order_models_by="ROC_AUC", 
    n=5, groupby="model_name"
    ):
    '''
        helper function, that extract "n" best performing models, 
        according to "order_models_by", in groups created with groupby
        based on fdata stored in "model_statistics_means"
    '''
    # group by model name or dataset name
    grp = model_statistics_means.groupby(by=groupby)

    # group by 
    top_models = None
    for name, group in grp:
        top_models_group = group.sort_values(by=order_models_by, ascending=False).iloc[0:n,:]
        if top_models is None:
            top_models = top_models_group
        else:
            top_models = pd.concat([top_models, top_models_group], axis=0)
            
    return top_models




# Function, ......................................................
def modified_model_stats_and_plot_ROC_curves(
    y, y_hat, y_hat_probs, model_fingerprint=dict(),
    plot_title="", figsize=(8,3), positive_class=1, negative_class=0,
    create_plot=False, tr=0.5, ID=None, 
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
    ns_probs = np.array([positive_class for _ in range(y.shape[0])])        
        
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
            "ID":ID,
            "model_name": model_fingerprint["model_name"],
            "dataset_name": model_fingerprint["dataset_name"],
            
            # basic stats    
            "Acc_baseline": accuracy_score(y, y_hat_baseline),
            "Acc": accuracy_score(y, y_hat),
            
            # diagnostic stats
            "ROC_AUC": roc_auc_score(y, y_hat_probs),
            "Recall": recall_score(y, y_hat),
            "Presision": precision_score(y, y_hat), 
            "F1": f1_score(y, y_hat),
        
            # examples
            "tr":tr,
            "counts_y": str(dict(zip(y_unique, y_counts))),
            "counts_y_hat": str(dict(zip(y_hat_unique, y_hat_counts))),
            
            # params
            "model_params": model_fingerprint["model_params"],
        }
    
    # Plot, ............................................. 
    if create_plot==True:
    
        # create plot with 2 subplots for ROC and PR curves
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.suptitle(plot_title)

        # . roc curve
        axs[0].plot(model_fpr, model_tpr, marker='.', label="MODEL")
        axs[0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        axs[0].scatter(default_model_fpr, default_model_tpr, s=1000, marker="o", color="red")
        axs[0].scatter(default_model_fpr, default_model_tpr, s=10, marker="o", color="red", label="therhold")
    
        # . precision-recall curves
        axs[1].plot(model_recall, model_precision, marker='.', label="MODEL")
        axs[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        try:
            axs[1].scatter(default_model_recall, default_model_precision, s=1000, marker="o", color="red")
            axs[1].scatter(default_model_recall, default_model_precision, s=10, marker="o", color="red", label="therhold")
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

    return pd.DataFrame(pd.Series(stat_results)).transpose(), roc_thresholds



# helper function, ...............................................
def play_with_rocks(
    ID, threshold, predictions, predictions_second_set=None, 
    display_available_thresholds=False, positive_class=1,
    title="Model Stats", subtitle="", class_description=None

):
    ''' helper function that allows plotting ROC cusves, of models just by slecteing their ID, 
        and providing df with model_predictions, 
        . ID; ID number visible in all summary tables, next to eahc model,
            you may just add this number here and play wiht thresholds to visualize the effects
        . predicitons- in this project, df with prediciton for validation data, created as subset of train data, 
            thus they have target varianble, 
        . prediciton_second_set; None or pandas df with predicitons for data provided wihtout target variable, called as test, 
            these are often provided without true label, that will be indicated as unknwonw
            if None provided, predicitons for main set are used to made this table, 
        . threshold; float, a custoff at y_hat_proba, for selecting class 1, 
        . title; str, title displayed on the top of the report
        . subtitle; str of None, subtitle displayed above mini-report for the predicitons_second_set
        . class_description; str, or dict, with description of classes
        
        caution, use it only with binary classiers
    '''
    
    # title
    print("---------------------------------------------------------------------------------------------------")
    print("TITLE: ", title)
    if class_description is not None:
        print("CLASSES: ", class_description)
    else:
        pass
    print("POSITIVE CLASS: ", positive_class)
    print("MODEL ID: ", ID)
    print("THRESHOLD: ", threshold)
    print("...............................................................................................")
    
    # (a) setup

    # . positive class
    if positive_class==0:
        negative_class=1
    else:
        positive_class=1
        negative_class=1

    # . get preddictions to fill in table, for a given model (ID)
    predictions_with_y = predictions[ID].copy()
    if predictions_second_set is None:
        predictions_without_y = predictions[ID].copy()
    else:
        predictions_without_y = predictions_second_set[ID].copy()
        
    # . create model fingepring 
    '''ie. basic info on the model stored in df with prediction'''
    model_fingerprint = {
        "model_name": predictions_with_y.loc[:, "model_name"].iloc[0],
        "dataset_name": predictions_with_y.loc[:, "dataset_name"].iloc[0],
        "model_params": str(predictions_with_y.loc[:, "model_params"].iloc[0]) # so there is no nested dictionaries
    }

    # . extract predictions_with_y
    y = predictions_with_y.loc[:,"y"].values.astype(int)
    y_hat = predictions_with_y.loc[:, "yhat"].values.astype(int)
    y_hat_probs = predictions_with_y.loc[:, "yhat_prob"].values.astype(float)

    # . extract predictions_without_y
    y_without_y = predictions_without_y.loc[:, "y"] # its often None, if none was provided for test dataset
    y_hat_without_y = predictions_without_y.loc[:, "yhat"]
    y_hat_probs_without_y = predictions_without_y.loc[:, "yhat_prob"]

    # (c) ROC curve plot

    # . provide everythign to my function
    model_report, roc_thresholds = modified_model_stats_and_plot_ROC_curves( 
        y=y, 
        y_hat=y_hat, 
        y_hat_probs=y_hat_probs, 
        model_fingerprint=model_fingerprint,
        plot_title=f'model {ID}\n{model_fingerprint["model_name"]}, {model_fingerprint["dataset_name"]}', 
        figsize=(8,3), 
        positive_class=positive_class, 
        negative_class=negative_class,
        create_plot=True, 
        tr=threshold
    )

    # (d) provide preditions with new threshold to test data
    new_predictions = [positive_class if x>=threshold else negative_class for x in y_hat_probs_without_y]
    new_prediciton_table = pd.DataFrame({
         "y_hat": new_predictions,
         "y_hat_probs": y_hat_probs_without_y,
         "trhreshold": [threshold]*len(y_hat_without_y)  
        })


    # (e) information 

    # . display the report under each plot
    print("...............................................................................................")
    print("MODEL PERFORMANCE")
    display(model_report)
    #model_report    

    # . test if you really made difference
    print("...............................................................................................")
    print("MODEL PREDICTIONS: ",subtitle)
    
    if y_without_y is not None:
        y_unique, y_counts = np.unique(y_without_y, return_counts=True)
        print("True labels:                                     ",dict(zip(y_unique, y_counts)))
    else:
        print("True labels:", "- UNKNOWN - ")
    
    y_unique, y_counts = np.unique(y_hat_without_y, return_counts=True)
    print("Predictions made with standard threshold (tr=0.5):", dict(zip(y_unique, y_counts)))
    y_unique, y_counts = np.unique(new_predictions, return_counts=True)
    print(f"Predictions made with adjusted threshold (tr={threshold}):", dict(zip(y_unique, y_counts)))
    
    if display_available_thresholds==True:
        print("...............................................................................................")
        display("AVAILABLE ROC THRESHOLDS",np.round(roc_thresholds,3)[1::])# donf use the first threshold
    else:
        pass
      
    print("---------------------------------------------------------------------------------------------------\n\n")

    # return 
    return new_prediciton_table