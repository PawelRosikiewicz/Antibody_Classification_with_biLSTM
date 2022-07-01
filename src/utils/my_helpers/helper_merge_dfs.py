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
import seaborn as sns # advance plots, for statistics, 
import matplotlib as mpl # to get some basif functions, heping with plot mnaking 
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt # for making plots, 


def merge_data(df1, df2, none_at=None, verbose=False):
    ''' vhelper function to join info in two dataframes, prepared in notebook 03
        in two planned dataset one of them will be missing, 
        df1 : data frame 1, or numpy array
        df2 : data frame 2, or numpy array
        none_at: None, it tries to merge df1, and df2, if 1, or 2, it will treat as respective df is missing
    '''
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    
    # check dimensions
    if verbose==True: 
        print("df1 shape: ", df1.shape)    
        print("df2 shape: ", df2.shape)    
    else:
        pass
    
    # ensure there is no reindexing
    df1.reset_index(inplace=True, drop=True)
    df2.reset_index(inplace=True, drop=True)
    
    # in case you wish to buold model only with one data type:
    if none_at is None: pass
    if none_at==1: df1=None
    if none_at==2: df2=None    
        
    # merge
    if df1 is not None and df2 is not None:
        final_dataset = pd.concat([df1, df2], axis=1)                                
    if df1 is not None and df2 is None:
        final_dataset = df1.copy()   
    if df1 is None and df2 is not None:
        final_dataset = df2.copy()
    
    # check final dimensions
    if verbose==True: 
        print("final df: ", final_dataset.shape)    
    else: 
        pass
    
    return final_dataset.values








