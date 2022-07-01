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

import numpy as np # support for multi-dimensional arrays and matrices
import pandas as pd # library for data manipulation and analysis
#import seaborn as sns # advance plots, for statistics, 
#import matplotlib as mpl # to get basic plt   functions, heping with plot mnaking 
#import matplotlib.pyplot as plt # for making plots, 



# Function, ...................................
def get_feature_summary(df, features, fillna=None):
    ''' helper function: provides nicely fomratted table 
        with counts and percenatge of each group defined with several features
        parameters:
        . df; pandas, data frame
        . features; list with 2 or 3 column names in df
        . fillna; None, or str/int, value for pandas.fillna(value=<fillna>)
        returns:
        . pandas data frame
        example:
        >>> features = ['target','Sex', 'Immune phenotype']
        >>> get_summary(data_temp, features, fillna="MISSING") 
    '''
    # test input df, & work on df copy,
    assert type(df) == pd.DataFrame, "Incorrect obj type"
    df = df.copy()
    
    # fillna
    if fillna is not None: 
        df = df.fillna(value=fillna)
    else:
        pass
    
    # group the data with provided features, 
    grp = df.groupby(by=features)
    # count instances in each group and store index
    grp_df = grp.size().unstack().fillna(0)
    grp_df_index = grp_df.index
    # calulate percentages
    grp_perc = grp_df.div(grp_df.sum(axis=1), axis=0)

    # reset indexes, to avoid problems with many selected features
    grp_df.reset_index(drop=True, inplace=True)
    grp_perc.reset_index(drop=True, inplace=True)
    
    # create nice output with counts, and percentage value per row
    'a user-fiendly format eg: 19 (52%)'
    df_res=[]
    for index in range(grp_df.shape[0]):
        df_res.append([f'{int(x)}   ({int(y*100)}%)' 
         for x, y in zip(
             grp_df.iloc[index,:].to_list(), 
             grp_perc.iloc[index,:].to_list()
         )])
    
    # "transplant" column names and indexes created by groupby
    df_res = pd.DataFrame(
        df_res, 
        columns=grp_df.columns
    )
    
    # add column with sum
    
    # empty column - for visual readibility
    empty_col = pd.Series(["-"]*df_res.shape[0])
    # row sums
    row_sums = grp_df.sum(axis=1)
    # add row percentage of total
    row_sums = pd.Series([f'{x} ({int(y)}%)' for x, y in zip(row_sums, row_sums/row_sums.sum()*100)])
    
    # concatenate all, 
    df_res = pd.concat([df_res, empty_col, row_sums],axis=1)
    
    # aestetics
    
    # add multilevel index
    df_res.index=grp_df_index
    # chnage two last colum labels
    df_res.columns.values[-2::]=["-","sum & (% of total)"]
    
    return df_res, grp_df