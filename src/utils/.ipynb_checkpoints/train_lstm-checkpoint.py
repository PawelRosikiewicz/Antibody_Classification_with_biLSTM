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

import os # allow changing, and navigating files and folders, 
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import random # functions that use and generate random numbers
import pickle

import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # advance plots, for statistics, 
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt # for making plots, 

# model traning and selection tools
from src.utils.Model_Evaluation_PIPELINE import load_pickled_data
from src.utils.data_preprocessing_pipeline import pickle_saver

# load ltsm related scripts
from src.utils.ablstm import ModelLSTM



# Function, ..................................................................
def subset_Xy(x, y, subset=None, rand_nr=0):
    ''' create smaller subset of the data to allow fast trianing
        and parameter selection, additionally it shuffles the data
        . X, y; numpy 2D and 1D array, repsectively,
        . subset; None, or float(0,1), if None, the data are just shuffled, in X,y in the same order
        
        Comment: is subset==None.then the data are just shuffled,  
    '''  
    # set seed
    np.random.seed(rand_nr)
    
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
    

    
# Function, ..................................................
def join_func(s):
    ''' helper funciton to join all aa/gaps 
        in aa-seq into one string used by lstm models
    '''
    return "".join(s.values.tolist())



# Function, ..................................................
def unpack_my_data(data_prep, subset=None, subset_all=True, shuffle=True, verbose=False):  
    ''' Loads and unpack pickled data, preparted with data preprocessing pipeline, stored in dictioraries,
        . data_prep; dct, wiht Xma dn y data with train, valid and test subsets, 
        . subset; bool, if True, test and valid are also subset and shuffled, 
        . subset_all; bool, if True, test and validation is also subsetted, 
        . shuffle; bool, if True, it imposes shuffle on train data, 
        
        Comment: if you wish to just shuffle the train data, no subsetting, no shuffle in test and validation
        use: "subset=None, subset_all=False, shuffle=True"
    '''

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



# Function, ..................................................
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



# Function, ..................................................
def check_labels(lab, s):
    '''helper function to chek if we selected corrrect labels'''
    uel = s.unique().tolist()
    if len(uel)==1 and uel[0]==lab: return "Labels are OK"
    if len(uel)==1 and uel[0]!=lab: return "Incorrect label"
    if len(uel)>1 and uel[0]==lab: return "multiple labels detected"
    
    

# Function, ..................................................
def create_LSTM_features(
    data_dct=None,
    file_name_in=None, file_name_out="tst_scores_dct.p", 
    path_in=None, path_out=None, class_description=None, subset=0.5, subset_all=True, 
    lstm_params=None, lstm_fit_params=None, random_nr=0, train_size=0.7, verbose=False):
    ''' trains lstm models for each class, and stores tst scores as features in new feature matrix
    
        parameters:
        . data_dct=None,
        . file_name_in=None, 
        . file_name_out; str, name of the file saved wiuth the results in path_out
        . path_in; st, full path to input data, if data_dct=None, othwerwise not used, 
        . path_out, str -||- for oputpout data saving
        . class_description; list wiht class names, to display, if None, 
        . subset; none of Float,<1, indicating proportion of samples selected randomly wihtout repeats  to reduce the dataset size
        . subset_all; bool, if True, all datasets including validaiton and test will be reduced in number of samples, selected randomly
        . lstm_params, ditc with parameters for ModelLSTM initialization, 
        . lstm_fit_params; dict wiht parameters for  ModelLSTM.fit() method
        . verbose; bool, 
    
        comments:
        . params examples:
        >>> lstm_params = dict(embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True)
        >>> lstm_fir_params = dict(n_epoch=1, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None)
    '''
    
    # setup path and params, ..........................................
    if lstm_params is None:
        lstm_params = dict(embedding_dim=64, hidden_dim=64, device='cpu', gapped=True, fixed_len=True)
    if lstm_fit_params is None:
        lstm_fit_params = dict(n_epoch=1, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None)
    if path_in is None:
        path_in = os.getcwd()
    if path_out is None:
        path_out = os.getcwd()    
    
    # data preparation .............................................
    ''' you can either load data from path, 
        or use dataset prepared directly with the pipeline'''

    if verbose==True: 
        if verbose==True: print("\nPART 1. DATA PREPARATION\n")
    else:
        pass

    # . setup dataset size
    if subset_all==True:
        subset_valid = subset
    else:
        subset_valid = 1
   
    if data_dct is None:
        # . load the data for LSTM model
        x_tr, y_tr, x_val, y_val, _, _ = load_pickled_data(
            path_in, file_name_in, 
            subset=subset, # Use only 0.1 data for trainingm data will be shuffled 
            subset_all=True, # test and valid data will also be reduced in size, 
            shuffle=True, 
            verbose=verbose
        )

        # . load full data for evaluation procedure
        _, _, x_valid, y_valid, x_test, y_test = load_pickled_data(
            PATH_data_interim, file_name_in, 
            subset=subset_valid, # no subset for predictions
            subset_all=True, # if True,  test and valid data will also be reduced in size, 
            shuffle=True, 
            verbose=verbose
        )
    else:
        # . load the data for LSTM model
        x_tr, y_tr, x_val, y_val, _, _ = unpack_my_data(
            data_dct, 
            subset=subset, # Use only 0.1 data for trainingm data will be shuffled 
            subset_all=True, # test and valid data will also be reduced in size, 
            shuffle=False, 
            verbose=verbose
        )

        # . load full data for evaluation procedure
        _, _, x_valid, y_valid, x_test, y_test = unpack_my_data(
            data_dct, 
            subset=subset_valid, # no subset for predictions
            subset_all=True, # if True, test and valid data will also be reduced in size, 
            shuffle=True, 
            verbose=verbose
        )        
        
    # check for class description
    if class_description is None:
        class_description = pd.Series(y_tr).unique().tolist()
    else:
        pass
        

    # LSTM ........................................................
    
    if verbose==True: print("\nPART 2. MODEL TRAINING\n")
    
    # use LSTMs to create new features
    for class_id, class_label in zip(pd.Series(y_tr).unique().tolist(), class_description):
        if verbose==True: print(f"Processing: {class_id}, {class_label}")

        # prepare the data fr model tranining 
        
        # (a) subset data for one class one for training
        trn_fn = pd.DataFrame(x_tr).loc[y_tr==class_id, :].apply(join_func, axis=1).values
        vld_fn = pd.DataFrame(x_val).loc[y_val==class_id, :].apply(join_func, axis=1).values
        
        # (b) work on copies for predictions
        x_valid_for_pred = x_valid.copy()
        x_test_for_pred = x_test.copy()
        
        # info
        if verbose==True: 
            print("\n. Data Used for LSTM model Training")
            print(f" - one class train data: {trn_fn.shape}, {check_labels(class_id, pd.Series(y_tr).loc[y_tr==class_id])}")
            print(f" - one class valid data: {vld_fn.shape}, {check_labels(class_id, pd.Series(y_valid).loc[y_valid==class_id])}")
            print("\n. Traning LTSM Model")
        else:
            pass
            
        # initialize model
        '''change device to 'cpu' if CUDA is not working properly'''
        model = ModelLSTM(**lstm_params)
        if verbose==True:  print(' - Model initialized.')

        # training
        model.fit(trn_fn=trn_fn, vld_fn=vld_fn, **lstm_fit_params)
        if verbose==True: print(' - Done')

        # predictions
        if verbose==True:  print('\n. Making Predictions')
        tst_scores_valid_one_class = model.eval(
            fn=pd.DataFrame(x_valid_for_pred).apply(join_func, axis=1).values,
            batch_size=512
        )
        tst_scores_test_one_class = model.eval(
            fn=pd.DataFrame(x_test_for_pred).apply(join_func, axis=1).values,
            batch_size=512
        )

        # create new feature matrix for classyfication models
        '''caution! samples must be in the same order, otherwise you will get useless results'''
        if class_id==0:
            tst_scores_valid = tst_scores_valid_one_class
            tst_scores_test = tst_scores_test_one_class
        else:
            tst_scores_valid = np.c_[tst_scores_valid, tst_scores_valid_one_class]
            tst_scores_test = np.c_[tst_scores_test, tst_scores_test_one_class]       

    # info
    if verbose==True: 
        print(' - Done')
        print(f'   tst data set shape for validation data is: {tst_scores_valid.shape}')
        print(f'   ie. one column per each class')
    else:
        pass
        
    # SAVE FILES ........................................................
    ''' I am using the same data storage format as in all other classificaiton task, 
        this way, I can use my other pipelines 
    '''
    
    if verbose==True: 
        print("\nPART 3. SAVING FILES\n")
    else:
        pass
    pickle_saver(
        tst_scores_valid, tst_scores_test, tst_scores_test, 
        y_valid, y_test, y_test,
        path=path_out, fname=file_name_out, 
        verbose=verbose
    )    
    
    return tst_scores_valid, y_valid, tst_scores_test, y_test
  
    
    
# Function, .................................................................................
def plot_scatter(ax, data, y, class_labels, colors, title):
    '''helper funciton to plot the scatter plots with two or more series 
       of data stored in the first two columns in numpy array
    '''
    for class_id, class_label in zip([0,1], class_labels):
        data_subset=data[y==class_id,:]
        ax.scatter(x=data_subset[:,0], y=data_subset[:,1], 
                   s=10, alpha=0.1, label=class_label, color=colors[class_label])
    ax.grid(lw=0.5, ls=":", color="grey")
    ax.set_title(title)
    ax.set_xlabel(f"{class_labels[0]} scores")
    ax.set_ylabel(f"{class_labels[1]} scores")
    sns.despine()
    ax.legend()
    
    
    
# Function, .................................................................................
def plot_hist(origine, ax, data, y, class_labels, colors, title):
    '''helper funciton to plot the scatter plots with two or more series 
       of data stored in the first two columns in numpy array
    '''
    for class_id, class_label in zip([0,1], class_labels):
        data_subset=data[y==class_id,:]
        ax.hist(data_subset[:,origine], alpha=0.2, label=class_label, 
                edgecolor=colors[class_label], color=colors[class_label], 
                histtype="stepfilled", linewidth=3)
    ax.grid(lw=0.5, ls=":", color="grey")
    ax.set_title(title)
    ax.set_xlabel(f"tst scores")
    ax.set_ylabel(f"Frequency")
    sns.despine()
    legend = ax.legend(frameon=False)    
    legend.set_title("aa-seq from:")