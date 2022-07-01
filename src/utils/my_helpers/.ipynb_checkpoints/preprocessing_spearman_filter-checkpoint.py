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



class SpearmanFilter():
    ''' .............................................................
        Custom made transformer for TMP data provided from RNAseq experiments
        .............................................................        
        Allows Rank-based filtering with Spearman correlation of samples on rnaseq data.
        * first it creates an average gene expression profile for all samples
        * then, it calulates speerman rho correlation coef, for each sample, and the mean profile
        * finally, it removes the samples, that are either below preset tthreshold, or at lower quantile, 
        * the threshold, methods, and averaging methods are parametrized  
        
        FUNCTIONS
        . fit_transform()    : see below fit transform policy
        . transform()        : see funciton for transfomed policy
    '''

    # (a) private methods and variables, 

    # Method,.....................................
    def __init__(self):
        
        # parametrs
        self._tr=None # its 1-speakerm corr. with the mean, sample 
        self._quantile=None
        self._method=None
        self._avg_menthod=None

        # train data
        self._train_mean_profile=None
        self._train_samples_corr=None
        self._train_samples_removed=None
        self._train_samples_accepted=None
        
        # test data
        self._test_samples_corr=None
        self._test_samples_removed=None
        self._test_samples_accepted=None        
    
    # (b) public methods
    
    # Method,.....................................
    def fit_transform(self, x, y, tr=0.95, quantile=True, method='spearman', avg_menthod="median"):
        ''' Allows Rank-based filtering with Spearman correlation of samples on rnaseq data.
            * first it creates an average gene expression profile for all samples
            * then, it calulates speerman rho correlation coef, for each sample, and the mean profile
            * finally, it removes the samples, that are either below preset tthreshold, or at lower quantile, 
            * the threshold, methods, and averaging methods are parametrized  
            
            parameters:
            . x; Pandas DataFrame
            . y; pandas Series
            . tr; float [0-1], if qunatile=False (see below) it will reject all samples we corr<tr, 
            . quantile; bool, if True, tr used to calulate lower quantile boudary for 1-tr 
            . method; str, eg: 'spearman', method from pandas.df.corr()
            . avg_menthod; str, "median", or "mean"
            
            returns:
            . trandsformed x; Pandas DataFrame
            . trandsformed y; pandas Series            
            
            comments:
            parameters used, and list of corr, results, and samples rejected and accepted are available as private variables, 
        '''
        
        # store threshold parameters,  
        self._tr=tr # may be modified later later, based on the quantile options
        self._quantile=quantile
        self._method=method
        self._avg_menthod=avg_menthod
        
        # Test input df,
        assert type(x) == pd.DataFrame, "Incorrect obj type: x shoudl be pd.Series"
        assert type(y) == pd.Series, "Incorrect obj type: y shoudl be pd.Series"

        # create average gen expression profile for train data
        if avg_menthod=="median":
            self._train_mean_profile = x.apply(np.median, axis=0)
        else:
            self._train_mean_profile = x.apply(np.mean, axis=0)
        train_mean_profile = self._train_mean_profile
            
            
        # calulate spearman corr between each sample, and avg_profile
        ''' done with for lopp to avoid any error, 
            and becuase i have relatively small sample nr
        '''
        train_samples_corr=[]
        for index in range(x.shape[0]):
            # calulate corr.
            dftemp = pd.concat([x.iloc[index,:], train_mean_profile], axis=1)
            one_sample_corr = dftemp.corr(method=method).iloc[0,1]

            # store the results
            train_samples_corr.append(one_sample_corr)

        # keep corr results, and sample Id's inside pd series,
        train_samples_corr = pd.Series(train_samples_corr, index=x.index.values.tolist())
        self._train_samples_corr = train_samples_corr

        # find samples to remove and accept
        if quantile==False:
            '''threshold is used dirently, to filter the samples with corr results'''
            self._train_samples_removed = train_samples_corr.iloc[(train_samples_corr<tr).values.tolist()].index.values.tolist()
            self._train_samples_accepted = train_samples_corr.iloc[(train_samples_corr>=tr).values.tolist()].index.values.tolist()
        else:
            '''1-threshold is used as quantile value'''
            lower_quntile_tr = train_samples_corr.quantile(1-tr)
            self._train_samples_removed = train_samples_corr.iloc[(train_samples_corr<lower_quntile_tr).values.tolist()].index.values.tolist()
            self._train_samples_accepted = train_samples_corr.iloc[(train_samples_corr>=lower_quntile_tr).values.tolist()].index.values.tolist() 
            self._tr=lower_quntile_tr

        # remove rejected samples and return the data
        x_transf = x.iloc[self._train_samples_accepted,:]
        y_transf = y.iloc[self._train_samples_accepted]
        
        return x_transf, y_transf 
            
            
    # Method,.....................................
    def transform(self, x, y, inform=False):
        ''' transform method for Rank-based filtering with Spearman correlation of samples on rnaseq data.
            * first it creates an average gene expression profile for all samples
            * then, it calulates speerman rho correlation coef, for each sample, and the mean profile
            * finally, it removes the samples, that are either below preset tthreshold, or at lower quantile, 
            * the threshold, methods, and averaging methods are parametrized  
            
            parameters:
            . x; Pandas DataFrame
            . y; pandas Series
            . inform; bool; if True, fucntion will return pd.series with correlation vvalues only, 
            
            returns:
            . trandsformed x; Pandas DataFrame
            . trandsformed y; pandas Series            
            
            comments:
            parameters used, and list of corr, results, and samples rejected and accepted are available as private variables, 
        '''
        # store threshold parameters,  
        tr = self._tr
        quantile = self._quantile
        method = self._method
        avg_menthod = self._avg_menthod
        train_mean_profile = self._train_mean_profile
        
        # Test input df,
        assert type(x) == pd.DataFrame, "Incorrect obj type: x shoudl be pd.Series"
        assert type(y) == pd.Series, "Incorrect obj type: y shoudl be pd.Series"

        # calulate spearman corr between each sample, and avg_profile
        ''' done with for lopp to avoid any error, 
            and becuase i have relatively small sample nr
        '''
        test_samples_corr=[]
        for index in range(x.shape[0]):
            # calulate corr.
            dftemp = pd.concat([x.iloc[index,:], train_mean_profile], axis=1)
            one_sample_corr = dftemp.corr(method=method).iloc[0,1]

            # store the results
            test_samples_corr.append(one_sample_corr)

        # keep corr results, and sample Id's inside pd series,
        test_samples_corr = pd.Series(test_samples_corr, index=x.index.values.tolist())
        self._test_samples_corr = test_samples_corr

        # find samples to remove and accept
        '''in tranfomr the threshold is taken from fit_transform method'''
        self._test_samples_removed = test_samples_corr.iloc[(test_samples_corr<tr).values.tolist()].index.values.tolist()
        self._test_samples_accepted = test_samples_corr.iloc[(test_samples_corr>=tr).values.tolist()].index.values.tolist()

        # remove rejected samples and return the data
        x_transf = x.iloc[self._test_samples_accepted,:]
        y_transf = y.iloc[self._test_samples_accepted]
        
        if inform==False:
            return x_transf, y_transf 
        else:
            return self._train_samples_corr, self._test_samples_corr