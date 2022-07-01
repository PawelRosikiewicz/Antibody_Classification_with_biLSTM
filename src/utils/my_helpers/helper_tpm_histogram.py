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
def tpm_hist(
    df,
    labels=None,
    randomstate=0,
    n=5,
    figsize=(15,2),
    remove_zeros=True, # remove values <=0
    density=False,
    avg_profile="class", # "global"
    colors = ("grey", "red"),
    title=""
    ):
    ''' draws step-type histogram for n examples (rows in input dataframe)
        if labes are provided, it draws histogram for n examples   
        
        parameters
        . df; pd.DataFrame, rows used to plot data on individual histograms, 
        . labels; pd.Series, or None, if series provided, n-.examples will be plotted on hist on different figures, 
        . randomstate; int, used to have the same selection on all plots
        . n; int, number of row examples, sampled, from each class to make the figure, each example is plotted on separate subplot 
        . figsize, tuple with two int, 
        . remove_zeros; if true, it removes remove values <=0
        . density; if True, areay under the hist sums to 1, 
        . avg_profile="class", # "global"
        . colors; tuple, with two colors, idx=0, for average  
        . title; str, added at the beginning of the title
        
        returns
        . fig or figures, with n-subplots with hist for each selected row, in input data frame. one figure is created for each class, in labels, 
    '''
    # set colors and hist params
    hist_params = dict(density=density, histtype="step", bins=100, linestyle="-")

    # set see to have the same selection on all plots
    np.random.seed(randomstate)

    # check for labels
    if labels is None:
        labels = pd.Series(["examples taken from all samples"]*df.shape[0])
    else:
        pass

    # get groups to plot on separate plots
    labels_unique = labels.unique().tolist()

    # creat one figure per group
    for i, label in enumerate(labels_unique):

        # find the group
        class_label_idx = np.arange(df.shape[0])[(labels==label)]

        # check if you have enought n, to smaple from
        if n<=len(class_label_idx):
            available_n = n
        else:
            available_n = len(class_label_idx)

        # select n examples, 
        selected_idx = np.random.choice(class_label_idx, size=available_n, replace=False)

        # create average profile for the class/ or for all samples, 
        if avg_profile=="class":
            avg_class_profile = df.iloc[class_label_idx,:].median(axis=0)    
        else:
            avg_class_profile = df.median(axis=0)   
        
        # remove zeros, 
        if remove_zeros==True:
            avg_class_profile = avg_class_profile.loc[avg_class_profile>0]
        else:
            pass

        # create figure
        fig, axs = plt.subplots(ncols=available_n, nrows=1, figsize=figsize, facecolor="white")
        plt.suptitle(f"{title} - class: {label}")

        # plot hist of gene expr. profile in each 
        for plt_i, ax in enumerate(axs.flat):

            # prepare the data
            data_for_hist = df.iloc[selected_idx[plt_i],:]
            if remove_zeros==True:
                data_for_hist = data_for_hist.loc[data_for_hist>0]
            else:
                pass

            #ax.hist(average_class_profile, bins=100)
            ax.hist(avg_class_profile, color=colors[0], label="average", linewidth=4, alpha=0.5, **hist_params)
            ax.hist(data_for_hist, color=colors[1], label=f"sample {selected_idx[plt_i]}", linewidth=1, **hist_params)

            # title. grid, and legen
            ax.set_title(f"sample nr: {selected_idx[plt_i]}\n- tpm>0: {(data_for_hist>0).sum()}\n- tpm=0: {df.shape[1]-(data_for_hist>0).sum()}", 
                         fontsize=8, ha="left", color="black")
            ax.legend(frameon=False, fontsize=8)

        fig.tight_layout()
        fig.subplots_adjust(top=0.6)
        sns.despine()
        plt.show();


