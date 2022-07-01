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

# my custom functions, 
from src.utils.eda_helpers import aa_seq_len_hist
from src.utils.eda_helpers import aa_seq_qc # QC table on loaded qq-seq data
from src.utils.qc_helpers import unique_aa_counts_hist



# Function, ..................................................
def create_aa_matrix(s, verbose=False):
    '''create dataframe with each aa in separate column
        . s; pd.Seires, with aa-sequences as strings,
        . verbose; bool, if True, returns shape of a new object
    '''
    # data preparation
    aa_mat = s.str.split(pat="",expand=True)
    # . remove artifact: enpty space at begining and end
    aa_mat = aa_mat.iloc[:,1::]
    aa_mat = aa_mat.iloc[:,:-1]
    
    if verbose==True:
        print("AA matrix shape: ", aa_mat.shape)
    else:
        pass

    return aa_mat



# Function, .............................................................       
def calc_aa_perc_per_pos(s):
    '''
        checks aa composition - eg to spot some irregularities
        used only with aligned sequences
        Caution: this is modified fucntion for ml, pipelines, data preparaiton step 
    '''
    aa_mat = create_aa_matrix(s, verbose=False)

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


  
# Function, .............................................................  
def calc_aa_number_per_pos(s, tr=None, min_nr=1):
    # check aa composition - eg to spot some irregularities
    ''' returns pd. series with number of different nuleotides or gap per position,
        . caution - use it only with aligned sequences,
        . tr; float, threshold for min % of aa/positon to be considered
        
        Caution: this is modified fucntion for ml, pipelines, data preparaiton step 
    '''
    # empty lists
    unique_aa_counts=[]
    unique_aa_perc=[]
    unique_aa_numbers=[]
    
    # data preparation
    aa_mat = create_aa_matrix(s, verbose=False)

    # create df, aa withcounts/position
    for col_idx in range(aa_mat.shape[1]):    
        # collect data for plot, 
        vc_per_pos_number = aa_mat.iloc[:,col_idx].value_counts()
        # . turn values into %
        vc_per_pos = vc_per_pos_number/aa_mat.shape[0]*100
        
        # apply % threshold
        if tr is not None:
            mask = vc_per_pos.copy()
            vc_per_pos = vc_per_pos.loc[mask>tr]
            vc_per_pos_number = vc_per_pos_number.loc[mask>tr]
        else:
            pass
    
        # apply min_nr threshold
        if min_nr is not None:
            mask = vc_per_pos_number.copy()
            vc_per_pos = vc_per_pos.loc[mask>min_nr]
            vc_per_pos_number = vc_per_pos_number.loc[mask>tr]
            vc_per_pos = vc_per_pos/vc_per_pos.sum()
        else:
            pass
        
        # store results
        unique_aa_perc.append(vc_per_pos)
        unique_aa_numbers.append(vc_per_pos_number)   
        unique_aa_counts.append(vc_per_pos.shape[0])        
        
    return unique_aa_counts, unique_aa_perc, unique_aa_numbers, aa_mat     



# function, .....................................................
def prepare_aa_data_for_baseline(X, y, tr=1, min_nr=2, shuffle_arr=False, verbose=False):

    for i, label in enumerate(np.unique(y).tolist()):
        # subset the data
        X_subset = X[y==label]

        # find which 
        unique_aa_counts, unique_aa_perc, unique_aa_numbers, aa_mat = calc_aa_number_per_pos(pd.Series(X_subset), tr=tr, min_nr=min_nr)
            
        # replace all aa below any threshold with most frequent, 
        for col_idx in range(aa_mat.shape[1]):
            # get aa on one positon for replacement
            one_position = pd.Series(aa_mat.iloc[:,col_idx]).copy()
            # . extarct info
            accepted_aa = unique_aa_perc[col_idx].index.values
            all_aa = one_position.unique().tolist()
            most_frequent_aa = accepted_aa[0]

            # aa below thresholds with most frequent aa
            if one_position.unique().size==accepted_aa.size:
                pass
            else:
                aa_replacement_dict = dict((x,x) if x in accepted_aa.tolist() else (x,most_frequent_aa) for x in all_aa)
                aa_mat.iloc[:,col_idx] = aa_mat.iloc[:,col_idx].map(aa_replacement_dict)   

        # build final df's for X, y
        if i ==0:
            data_arr = aa_mat
            y_to_return = pd.Series([label]*aa_mat.shape[0])
        else:
            data_arr = pd.concat([data_arr, aa_mat], axis=0)
            y_to_return = pd.concat([y_to_return, pd.Series([label]*aa_mat.shape[0])])
            # work on indexes, 
            data_arr.reset_index(drop=True, inplace=True)
            y_to_return.reset_index(drop=True, inplace=True)
                
        # qc on selection with histograms,
        if verbose==True:
            print("\nX.shape: ", data_arr.shape)
            print("y.shape: ", y_to_return.shape)
            unique_aa_counts_hist(unique_aa_counts, label, class_size=X_subset.shape[0], figsize=(3,3))
        else:
            pass
        
    # use from sklearn.utils import shuffle    
    if shuffle_arr==True:
        idx = shuffle(np.arange(data_arr.shape[0]))
        data_arr = data_arr.iloc[idx,:]
        y_to_return = y_to_return.iloc[idx]
        # work on indexes, 
        data_arr.reset_index(drop=True, inplace=True)
        y_to_return.reset_index(drop=True, inplace=True)        
    else:
        pass
                    
    return data_arr, y_to_return
  
  
  

