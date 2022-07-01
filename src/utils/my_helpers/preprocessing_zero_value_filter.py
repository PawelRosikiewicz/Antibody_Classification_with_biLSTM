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
import matplotlib as mpl # to get basic plt   functions, heping with plot mnaking 
import matplotlib.pyplot as plt # for making plots, 




class ZeroValueFilter():
    ''' .............................................................
        Custom made transformer for TMP data provided from RNAseq experiments
        .............................................................        
        removes columns representing genes wiht zero values, 
        or values below certain threshold defined by the user
        
        FUNCTIONS
        . fit_transform()    : see below fit transform policy
        . transform()        : see funciton for transfomed policy
    '''
    # (a) private methods and variables, 

    # Method,.....................................
    def __init__(self):
        
        # parametrs
        self._na_tr=None # its 1-speakerm corr. with the mean, sample 
        
        # train data (fit)
        self._train_idxcol_removed=None
        self._train_idxcol_accepted=None   
    
        # for qc 
        self._train_ZeroPerc_all=None # pd.Sseries, per gene % over samples
        self._train_ZeroPerc_accepted=None # pd.Sseries, per gene % over samples
        self._train_ZeroPerc_removed=None # pd.Sseries, per gene % over samples
        # ..
        self._test_ZeroPerc_all=None # pd.Sseries, per gene % over samples
        self._test_ZeroPerc_accepted=None # pd.Sseries, per gene % over samples
        self._test_ZeroPerc_removed=None # pd.Sseries, per gene % over samples

    # (b) public methods
    
    # Method,.....................................
    def fit_transform(self, x, na_tr=0.9):
        ''' all columns with % of values below 0 >tr, will be removed
            their indexes, are stored in the class
            
            parameters:
            . x; Pandas DataFrame
            . tr; float [0-1], if qunatile=False (see below) it will reject all samples we corr<tr, 
                important: if you set tr=0, it will remove columns with any number of zeros, 
                if you set 1, all columns are kept, irrspectively how much zero values it has, 
            . method; 
            
            returns:
            . trandsformed x; Pandas DataFrame         
        '''
        
        # store threshold parameters,  
        self._na_tr=na_tr

        # Test input and work on copy
        assert type(x) == pd.DataFrame, "Incorrect obj type: x shoudl be pd.Series"
        x = x.copy()

        # (b) fit
        
        # find how much of missing data you see in each gene
        NaPerc_per_col = (x==0).sum(axis=0)/x.shape[0]*100

        # find idx. of genes to remove
        mask = (NaPerc_per_col>(na_tr*100)).values.tolist()
        train_idxcol_removed = np.arange(x.shape[1])[mask].tolist()

        # find idx. of genes not to remove
        train_idxcol_accepted = pd.Series(np.arange(x.shape[1])).drop(train_idxcol_removed).values.tolist()

        # (b) save the results for training and qc, 

        # .. for transforming new data
        self._train_idxcol_accepted=train_idxcol_accepted 
        self._train_idxcol_removed=train_idxcol_removed
 
        # .. for qc
        self._train_ZeroPerc_all=NaPerc_per_col
        self._train_ZeroPerc_accepted=NaPerc_per_col.values[train_idxcol_accepted]
        self._train_ZeroPerc_removed=NaPerc_per_col.values[train_idxcol_removed]     

        # (c) transform & return
        x = x.iloc[:,train_idxcol_accepted]        
        
        return x
        
    # Method,.....................................
    def transform(self, x):
        ''' all columns with % of values below 0 >tr, will be removed
            their indexes, are stored in the class
            
            parameters:
            . x; Pandas DataFrame
            . tr; float [0-1], if qunatile=False (see below) it will reject all samples we corr<tr, 
            . method; 
            
            returns:
            . trandsformed x; Pandas DataFrame         
        '''
        
        # Test input and work on copy
        assert type(x) == pd.DataFrame, "Incorrect obj type: x shoudl be pd.Series"
        x = x.copy()

        # (a) find how much of missing data you see in each gene
        NaPerc_per_col = (x==0).sum(axis=0)/x.shape[0]*100

        # (b) save the results for training and qc, 
        self._test_ZeroPerc_all=NaPerc_per_col
        self._test_ZeroPerc_accepted=NaPerc_per_col.values[self._train_idxcol_accepted]
        self._test_ZeroPerc_removed=NaPerc_per_col.values[self._train_idxcol_removed]     

        # (c) transform & return
        x = x.iloc[:,self._train_idxcol_accepted]        
        
        return x