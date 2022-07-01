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

import matplotlib.gridspec
from scipy.spatial import distance
from scipy.cluster import hierarchy
from matplotlib.font_manager import FontProperties
from scipy.cluster.hierarchy import leaves_list, ClusterNode, leaders
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import manifold


# -----------------------------------------------
def pca_plot_and_scree_plot(x, y, figsize=(12,4), title="pca & scree plot", scale="n"):
    '''simple pca plot and scree plot with cumulative variance
        . x, y, pd. DataFrame and pd.Series, respectively,
        . title; str, for figure plot
        . figsize; tuple with two int
    '''
    # (a) DATA PREPARATION ...................    
    
    # work on copy
    x = x.copy()
    
    # scale
    if scale=="y":
        # ... Create Standard scaler
        scaler = StandardScaler()
        # ...Rescale data
        x_sc = scaler.fit_transform(x) # Use scaler.transform(X) to new data!
        x = pd.DataFrame(x_sc, columns=x.columns, index=x.index)
    else:
        pass
    
    # Compute first principal components,
    pca = PCA(n_components=None)
    
    # Compute component scores for x
    pca.fit(x);
    x_components  = pca.transform(x)    

    # Get Proportion of variance explained
    pve = pca.explained_variance_ratio_
        
    # labels
    y_labels      = y.copy()
    y_labels_unique = pd.Series(y_labels).unique().tolist()

    # (b) PCA PLOT ...........................   
        
    # create a figure:
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=figsize)
    plt.suptitle(title)

    # plot 1&2, and 2&3d axis 
    for i in [0,1]:
        ax = axs[i]
        # plot each label group separately
        for label in y_labels_unique: # 0,1,2,3..9

            # Images of this digit
            idx = (y_labels  == label)

            # Plot images
            ax.scatter(
                x_components[idx, 0], x_components[idx, 1+i],
                # Take a look at https://stackoverflow.com/a/17291915/3890306
                marker="h",
                s=100, # Size of each marker
                alpha=0.5,
                label=label
            )

        # ... Add legend and labels
        ax.set_title(f'PC{1}~PC{2+i}')
        ax.set_xlabel(f'PC{1}')
        ax.set_ylabel(f'PC{2+i}')
        ax.legend()

    # (b) PCA SCREE PLOTS ...................
    ax = axs[2]
    
    # ... Get Proportion of variance explained
    pve = pca.explained_variance_ratio_
    
    # ... Calculate cumulative sum
    pve_cumsum = np.cumsum(pve)
    
    # ... barplot, with varinace explained by each p. component
    xcor = np.arange(1, len(pve) + 1) # 1,2,..,n_components
    ax.bar(xcor, pve)
    #plt.xticks(xcor)

    # Add trendline with cumsum for variance explained by the model,using growing number of components

    # ... Add cumulative sum
    ax.step(
        xcor+0.5, # 1.5,2.5,..,n_components+0.5
        pve_cumsum, # Cumulative sum
        label="cumulative sum of \nvariance explained\nby the model\nusing growing nr\nof components",
        color="red"
    )

    # ... Add labels to the figure
    ax.set_title("scree plot")
    ax.set_xlabel('principal component')
    ax.set_ylabel('proportion of variance explained')
    ax.set_xlim(0,100)
    ax.legend()

    # ESTETICS .....................................    
    for i in range(3):
        ax = axs[i]
        
        # ... y-axis lebels number
        locator=plt.MaxNLocator(prune='both', nbins=10)
        ax.yaxis.set_major_locator(locator)

        # ... x.axis label nr
        locator=plt.MaxNLocator(prune='both', nbins=10)
        ax.xaxis.set_major_locator(locator)        

        # Remove ticks, and axes that you dot'n want, format the other ones,
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.spines['top'].set_visible(False) # remove ...
        ax.spines['right'].set_visible(False) # remove ... 

        # Format ticks,
        ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
        ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
        ax.yaxis.set_ticks_position('left')# shows only that
        ax.xaxis.set_ticks_position('bottom')# shows only that
        ax.tick_params(axis='x', rotation=70)

        # ... grid ...
        ax.grid(color="lightgrey", lw=0.5, ls=":")    

    plt.tight_layout()
    plt.show();

    
# -----------------------------------------------
# t-SNE embedding - has only fit_transform !
def tsne_plot(x, y, n_components=2, title="tsne plot", figsize=(5,4)):
    '''simple pca plot for 2 classes
        I am keeping small number of trasin dataset, 
        becuase that method takes a lot of time, 
        . x, y, pd. DataFrame and pd.Series, respectively,
        . n_components, int, n componenets for manifold.TSNE()
        
        comments:
        Use digits as markers  https://stackoverflow.com/a/17291915/3890306
    '''

    # labels
    y_labels      = y.copy()
    y_labels_unique = pd.Series(y_labels).unique().tolist()
    
    # t-SNE embedding of the MNIST digits dataset
    tsne   = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(x)

    # create a figure:
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    plt.suptitle(title)
    
    # ... Group points by label
    for label in y_labels_unique : # 0,1,2,3..9
        # find idx of points to plot
        idx = (y_labels == label)

        # add points
        ax.scatter(
            X_tsne[idx, 0], X_tsne[idx, 1],
            marker="h".format(label),
            s=100, # Size of each marker
            alpha=0.5,
            label=label
        )

    # ... Add legend and labels
    ax.set_xlabel('1st component')
    ax.set_ylabel('2nd component')
    sns.despine()
    ax.grid(lw=0.3)
    ax.legend()
    plt.show();    



