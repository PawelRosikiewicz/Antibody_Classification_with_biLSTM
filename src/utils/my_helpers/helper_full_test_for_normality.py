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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from src.utils.DataFrameExplorer.DataFrameExplorer_distribusion import normality_plots, check_normality, feature_distribution


# Function, .............................
def test_normality(df, fnames):
    ''' my function used to test normality in small number of features, eg. 1-6 features,
    
        Parameters:
        . df; pandas, data frame
        . features; list with 2 or 3 column names in df
        
        returns:
        . displays data frame, with results of statistical test for normality
        . shows plot, with "hist", "box", "probplot" of each feature
        . displays data frame wiht examples, from each feature
        
        Comments:
        if more features are tested I recommend using each funciton separately, 
        >>> from src.utils.DataFrameExplorer.DataFrameExplorer_distribusion import normality_plots, check_normality, feature_distribution
    '''
    
    
    # test input df, & work on df copy,
    assert type(df) == pd.DataFrame, "Incorrect obj type"
    df = df.copy()

    # 
    try:
    
        # use check_normality() function,for checking normality in each quantitative variable
        '''it returns dataframe with several popular statistics on each attribute'''
        pVal_table = check_normality(df=df, names=fnames)

        # display results
        print(f'\n{"".join(["."]*60)}\n- check normality wiht popular statistical tests\n{"".join(["."]*60)}')
        display(pVal_table)

        # create plots, ...............

        # information 
        print(f'\n{"".join(["."]*60)}\n-  check normality wiht plots and descriptive stats.\n{"".join(["."]*60)}')
        # create plots
        for plot_type in ["hist", "box", "probplot"]:   
            normality_plots(
                df=df, 
                plot_type=plot_type,
                names=fnames,
                figscale=1,
                distplot_dct={
                    "norm_hist":True, "rug":True, "kde":False , 
                    "fit":stats.norm,"hist_kws":{"linewidth":0.5}}
            )

        # provide examples from each feature.....

        # information
        print(f'\n{"".join(["."]*60)}\n- feature examples\n{"".join(["."]*60)}')
        # display table wiht examples
        df = dfe_features.show_df_exaples(df, fnames, n=3)
        display(df)
        
    except:
        print("Error: please, check dtype of the selected variables, or feature names provided")