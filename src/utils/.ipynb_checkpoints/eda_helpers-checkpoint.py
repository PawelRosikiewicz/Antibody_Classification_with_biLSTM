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

import os # allow changing, and navigating files and folders, 
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import random # functions that use and generate random numbers

import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # advance plots, for statistics, 
import matplotlib as mpl # to get basic plt   functions, heping with plot mnaking 
import matplotlib.pyplot as plt # for making plots, 

import matplotlib.patches as mpatches
from sklearn.utils import shuffle


# qc helper, .............................................................
def aa_seq_len_hist(s, title, color="forestgreen", limits=(0, 200)):
    ''' helper function for ploting hist with 
        aa seq lenght from pd.Series "s", and filename "title"
    '''
    # info text
    full_title = (f'{title}\nmean:, {np.round(s.str.len().mean(),2)}±{np.round(s.str.len().std(),2)}')
    # hist
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    fig.suptitle(full_title)
    
    ax.hist(s.str.len(),color=color, edgecolor=color, histtype="stepfilled", alpha=0.5, linewidth=5);
    ax.set_xlabel("aa sequence lenght")
    ax.set_ylabel('frequency')
    ax.set_xlim(limits)
    sns.despine()
    ax.grid(ls=":", lw=0.5)
    fig.subplots_adjust(top=0.7)
    plt.show();


    
# qc helper, .............................................................
def aa_seq_qc(s, title):
    '''helper funciton to help you displaying basic info on pd. series wiht aa sequences'''
    if title is None:
        title = "AA sequences"
    else:
        pass
    print("\n-----------------------------------------------")
    print(f"{title}")
    print("-----------------------------------------------")
    # file chekup:
    print(". shape:",s.shape)
    print(". unique el:", s.unique().shape)
    print(". non-unique el:", s.shape[0]-s.unique().shape[0])
    print(f'. max nr of duplicates: {[None if x==1 else x for x in [s.value_counts().values.max()]][0]}')
    print(f'. mean seq len:, {np.round(s.str.len().mean(),2)}±{np.round(s.str.len().std(),2)}')
    
    # show explaes
    print(". examples")
    display(s.head(2))

    
    
# loader, .............................................................
def load_aa_sequences(path, filename, color="forestgreen", verbose=False):
    '''loads txt files with aa sequences as pd.Series,
       using extension list of choice eg: .txt, vlen.txt etc...
       returns, files, in the same order as in the extension list
       . path; str
       . filename; str
       . verbose, bool, if True, function provides basic info on loaded file (shape, nr of unique seq, and their mean leneght)
       plus it display examples, df.head() funciton, 
       
    '''
    path_to_file = os.path.join(path, filename)
    # check if file exist, and load as pd.series, without header, 
    if os.path.exists(path_to_file):
        s = pd.read_table(path_to_file, header=None).iloc[:,0]
        if verbose==True: 
            aa_seq_qc(s, filename)
            aa_seq_len_hist(s, filename, limits=(0, 200), color=color)
        else: pass
        return s
    else:
        if verbose==True: 
            print(f"ValError: file not found: {path_to_file}")
 


# loader, .............................................................
def Load_data_for_ml(
    path,
    file_types,
    filenames_to_use,
    test_size = 0.15,
    verbose=True
    ):
    """ function, that loads all the data, from mouse aqnd human datasets
        join them in one dataframe, with the target, 
        shuffle the data
        and finally, it creates test data, to be used for making predicitons with selected model, 
        . path; str
        . file_types; list, with str, indicating file origine, eg. human
        . filenames_to_use; a dist, with key==file_types, and values; list, with files names to load on path,
        . test_size; float 0-1, franction of the data,. to be subset for validaiton dataset
        . verbose; bool, if True, it check dimensions and target composition in each subset, 
    """
    
    # for info
    print("-------------------------------------------------------------")
    
    # perform showrt qc on your sequences, 
    for i, file_type in enumerate(file_types):  
        for j, fname in enumerate(filenames_to_use[file_type]):

            # load aa-seq data
            s = load_aa_sequences(path, fname) 

            # . add labels
            s_labelled = pd.concat([s, pd.Series([i]*s.shape[0])], axis=1)
            s_labelled.columns=["aa_seq", "target"]            

            # . create one final dataset
            if i==0 and j==0:
                data =  s_labelled     
            else:
                data=pd.concat([data,  s_labelled], axis=0)
                data.drop_duplicates(keep='first', inplace=False)
                data.reset_index(drop=True, inplace=True)
            
            # info
            if verbose==True: 
                print("loaded: ", i, j, s.shape, file_type," - ", fname)
            else:
                pass 
    # info
    if verbose==True: 
        print("total dimension: ",data.shape, " - demultiplexed, and with target")
    else:
        pass     
            
    # shuffle
    "from sklearn.utils import shuffle "
    data = shuffle(data) 
    data.reset_index(drop=True, inplace=True)      

    # create train/validation and test subsets
    test_idx = int(np.ceil(data.shape[0]*test_size))
    data_test = data.iloc[0:test_idx,:]
    data_train_valid = data.iloc[test_idx::,:]

    # info
    if verbose==True:
        print("-------------------------------------------------------------")
        print(". train/validation data: ", data_train_valid.shape)
        print(data_train_valid.target.value_counts())
        print("\n. test data: ", data_test.shape)
        print(data_test.target.value_counts())
        print("-------------------------------------------------------------")
    else:
        pass

    return data_train_valid, data_test



  
# eda helper, .............................................................       
def calc_aa_perc_per_pos(s):
    # check aa composition - eg to spot some irregularities
    '''use only with aligned sequences'''
    aa_mat = s.str.split(pat="",expand=True)

    # remove artifact: enpty space at begining and end
    aa_mat = aa_mat.iloc[:,1::]
    aa_mat = aa_mat.iloc[:,:-1]

    # create df, aa withcounts/position
    for col_idx in range(aa_mat.shape[1]):
        if col_idx==0:
            vc = aa_mat.iloc[:,col_idx].value_counts()
        else:
            vc = pd.concat([vc, aa_mat.iloc[:,col_idx].value_counts()], axis=1)
        #print(col_idx, vc.shape)

    # replace NaN with 0 counts
    vc[vc.isnull()]=0

    # work on proportions, 
    vc = vc/aa_mat.shape[0]*100

    return vc            


  
# eda helper, .............................................................       
def calc_aa_number_per_pos(s, tr=None):
    # check aa composition - eg to spot some irregularities
    ''' returns pd. series with number of different nuleotides or gap per position,
        . caution - use it only with aligned sequences,
        . tr; float, threshold for min % of aa/positon to be considered
    '''
    # empty lists
    unique_aa_counts=[]
    unique_aa_perc=[]
    
    # data preparation
    aa_mat = s.str.split(pat="",expand=True)

    # . remove artifact: enpty space at begining and end
    aa_mat = aa_mat.iloc[:,1::]
    aa_mat = aa_mat.iloc[:,:-1]

    # create df, aa withcounts/position
    for col_idx in range(aa_mat.shape[1]):
        
        # how many unique aa is at each positon (or gaps)
        # unique_aa_counts.append(aa_mat.iloc[:,col_idx].unique().shape[0])
    
        # collect data for plot, 
        vc_per_pos = aa_mat.iloc[:,col_idx].value_counts()
        # . turn values into %
        vc_per_pos = vc_per_pos/aa_mat.shape[0]*100
        # apply % threshold
        if tr is not None:
            vc_per_pos = vc_per_pos.loc[vc_per_pos>=tr]
        else:
            pass
        # . store in a list
        unique_aa_perc.append(vc_per_pos)
        
        # how many unique aa is at each positon (or gaps)  
        unique_aa_counts.append(vc_per_pos.shape[0])
        

    return unique_aa_counts, unique_aa_perc      




