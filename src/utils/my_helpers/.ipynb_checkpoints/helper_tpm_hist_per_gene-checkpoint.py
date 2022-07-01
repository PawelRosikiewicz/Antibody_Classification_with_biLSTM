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



# Function, .......................................................
def tpm_hist_per_gene(
    df,
    genes=None, # list of genes, provided externally, to have the same genes observed on all plots,
    figsize=(15,2),
    remove_zeros=False, # remove values <=0
    density=False,
    colors = ("red", "forestgreen"),
    title=""
    ):
    ''' draws step-type histogram for n examples (rows in input dataframe)
        if labes are provided, it draws histogram for n examples   
        
        parameters
        . df; pd.DataFrame, rows used to plot data on individual histograms, 
        . genes; list, wiht genes to display on subplots
        . figsize, tuple with two int, 
        . remove_zeros; if true, it removes remove values <=0
        . density; if True, areay under the hist sums to 1, 
        . colors; tuple, with two colors, idx=0, sirst colors used for median, 
        . title; str
        
        returns
        . fig or figures, with n-subplots with hist for each selected row, in input data frame
    '''
    # data prep.
    
    # work on copy, and transpose
    '''I assume that genes are columns, and rows are samples'''
    assert type(df) == pd.DataFrame, "Incorrect obj type"
    df = df.transpose().copy()

    # gene gene names from data frame
    gene_names = df.index.values.tolist()
    
    
    # hist
        
    # create figure
    fig, axs = plt.subplots(ncols=len(genes), nrows=1, figsize=figsize, facecolor="white")
    plt.suptitle(f"{title}")

    # plot hist of gene expr. profile in each 
    for plt_i, ax in enumerate(axs.flat):

        # prepare the data
        idx = np.arange(df.shape[0])[pd.Series(gene_names)==genes[plt_i]]
        data_for_hist = df.iloc[idx[0],:]
        if remove_zeros==True:
            data_for_hist = data_for_hist.loc[data_for_hist>0]
        else:
            pass
        
        #ax.hist(average_class_profile, bins=100)
        hist_params = dict(density=density, histtype="step", bins=100, linestyle="-")
        ax.hist(data_for_hist, color=colors[1], label=f"samples", linewidth=1, **hist_params)
        ax.axvline(data_for_hist.median(), color=colors[0], label="median", linewidth=4, alpha=0.5)
            
        # title. grid, and legen
        ax.set_title(f"{genes[plt_i]}", fontsize=8, ha="center", color="black")
        ax.legend(frameon=False, fontsize=8, loc="best")# loc="upper right")

    fig.tight_layout()
    fig.subplots_adjust(top=0.6)
    sns.despine()
    plt.show();

