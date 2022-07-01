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

import matplotlib.patches as mpatches


# Function, ................
def spearman_clustermap(df, labels,  n=None, figsize=(10,10), title=None, cmap='RdBu', row_cmap='hsv', labels_2=None, row_cmap_2="binary"):
    ''' creates sns.clustermap, for all provided columns, with spearman corr.
        adds color labels to clustered features, 
        . df; pandas dataframe
        . labels; pd.series wiht len==df.shape[1], ie groups compared, 
        . labels2; -||-
        . n; number of subsapled idx's from data frame, 
        . figsize; tuple, with two int, 
        . cmap; seaborn cmap, 
        . title; str, 
        . cmap; cmap for heatmap, 
        . row_cmap; default, 'tab10', creates colors for rows, indicating class labels, 
        . row_cmap_2; default, 'tab10', creates colors for rows, indicating class labels, 
        
        comments:
        cmaps: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html
        'Diverging'; ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
        
    '''    
    # subset df columns, eg sample 1000 columns out of 10000
    if(n is None):
        df = df.copy()
    else:    
        idx = np.random.choice(list(range(df.shape[0])), size=n, replace=False)
        df = df.iloc[idx,:].copy()    
    
    # calculate correlations, 
    correlations = df.corr(method='spearman')

    # set row colors,optional
    if (labels is not None):
        labels_unique = labels.unique()
        lut = dict(zip(labels_unique, sns.color_palette(row_cmap, labels_unique.shape[0])))
        row_colors = labels.map(lut)
        
        # second layer labels, optional
        if (labels_2 is not None):
            labels_unique_2 = labels_2.unique()
            lut_2 = dict(zip(labels_unique_2, sns.color_palette(row_cmap_2, labels_unique_2.shape[0])))
            row_colors_2 = labels_2.map(lut_2)
            rc = pd.concat([row_colors_2, row_colors], axis=1) # the last one is displayed on the right site
            # and for the legend
            lut = {**lut, **lut_2}
        else:
            rc = row_colors     
    
    else:
        rc = None

    # clustermap
    g = sns.clustermap(correlations, method="complete", cmap=cmap, annot=False, 
                    vmin=-1, vmax=1, figsize=figsize, row_colors=rc)
    
    # add title, and axis labels, 
    if(title is not None):
        g.fig.suptitle(title) 
    else:
        pass
    # xaxis dendrogram,
    # g.fig.axes[2].set_ylabel("samples", fontsize=20)
    # heatmap,
    g.fig.axes[3].set_xlabel(f"samples")
    # small histogram legend,
    g.fig.axes[4].set_title("Spearman Corr.")    
    
    # Legend - later on, I wish to modify that part, 
    
    # create patch for each dataclass
    patch_list_for_legend =[]
    count_items = 0
    for i, cl_name in enumerate(list(pd.Series(lut).index.values)):
        cl_color = lut[cl_name]
        label_text = f"{cl_name}"
        patch_list_for_legend.append(mpatches.Patch(color=cl_color, label=label_text))
  
    # add patches to plot,
    l = g.fig.legend(handles=patch_list_for_legend, 
        loc="center", frameon=True, 
        scatterpoints=1, ncol=len(patch_list_for_legend), bbox_to_anchor=(0.5, 0.9), fontsize=10)
    
    # legend title - i left it here for future dev.
    # l.get_title().set_fontsize('10')
    # l.set_title(f'') 
            
    plt.show();