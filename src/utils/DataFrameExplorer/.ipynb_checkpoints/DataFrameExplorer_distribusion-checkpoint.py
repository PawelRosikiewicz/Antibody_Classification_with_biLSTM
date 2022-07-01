# ********************************************************************************** #
#                                                                                    #
#   Project: Data Frame Explorer                                                     #                         
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz(a)gmail.com                                                #
#                                                                                    #
#   License: MIT License                                                             #
#   Copyright (C) 2021.11.28 Pawel Rosikiewicz                                       #
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



# -*- coding: utf-8 -*-
import os
import sys
import re
import glob
import random
import itertools
import pathlib

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats




# Function, ........................................................
def normality_plots(
    df, names, 
    plot_type="hist",
    figsize=None, 
    figscale=1, nrows=1, 
    color=None, cmap='tab10', 
    title="", 
    distplot_dct={},
    boxplot_dct={}
):
    ''' ................................................................
        Returns nice looking histograms with sns.distplot function, 
        organized as subplots, 
        ................................................................
        parameters
        - df;  dataframe or numpy array,
        - feature_names; list, len==df.shape[1]
        _ plot_type: str, if hist: returns sns.displots with histogram,
                          if box, returns boxplots, for the same values in the same order,
                          if probplot, returns probability plots from scipy.stats
        - figscale; float, def==1, affect figure size, and fonsizes, 
        - nrows; int, def==1, how many rows to use, for subplots,  
        - color; str, or None, if str value provided, all histograms will have that color, 
        - cmap, str, matplotlib cmap, 
        - title; str, use for fig.suptitle()
        - distplot_dct; dictionary, with parameters for sns.displot()
        
        returns
        - matplotlib figure, 
    '''

    # work on copy of df subset
    assert isinstance(df, pd.core.frame.DataFrame), "df must be dataframe"
    df = pd.DataFrame(df.loc[:,names].copy()).copy()
    
    # rows/cols number
    nrows=nrows 
    ncols=int(np.ceil(len(names)/nrows))  
    
    # figsize
    if pd.isnull(figsize):
        figsize=(len(names)*3*figscale/nrows, 2*figscale*nrows)
    else:
        pass
    
    # prepare colorset, - can be used as cmap, 
    #.   cmap = mpl.cm.get_cmap('tab10')
    #.   rgba = cmap(np.linspace(0,1,len(names)))
    if pd.isnull(color):
        colors = sns.color_palette(cmap, len(names))
    else:
        colors = [color]*len(names)
    
    # Figure, 
    fig =plt.figure( figsize=figsize, facecolor="white")
    fig.suptitle(title, fontsize=10*figscale)
    
    # add each subplot separately,
    '''done this way, because other axs.flat doent work well with sns objects'''
    ax_number = 0 
    for irow in range(nrows):
        for icol in range(ncols):
            ax_number +=1
            
            if ax_number > len(names): 
                # no plot, but keep empty space
                break
                
            else:
                # add subplot
                ax = fig.add_subplot(int(nrows), int(ncols), int(ax_number))

                # create different type of plots 
                if plot_type=="hist":

                    # prep values displayed iunder the name
                    skewness = np.round(df.iloc[:,ax_number-1].skew(), 3)
                    kurtosis = np.round(df.iloc[:,ax_number-1].kurtosis(), 3)
                    skewness, kurtosis
                    
                    # histogram
                    ax = sns.distplot(            
                        df.iloc[:,ax_number-1], 
                        color=colors[ax_number-1],
                        **distplot_dct
                    )
                    ax.set_title(
                        f"{names[ax_number-1]}\nskewness {np.round(skewness,3)}\nkurtosis {np.round(kurtosis,3)}",
                        
                        fontsize=12*figscale)
                    ax.set_xlabel(f"{names[ax_number-1]}", fontsize=8*figscale)
                    ax.set_ylabel("Frequency", fontsize=8*figscale)
    
                elif plot_type=="box":
                    sns.boxplot(
                        data = df.iloc[:,ax_number-1],
                        ax = ax, 
                        orient="h",
                        color=colors[ax_number-1],
                        linewidth = 1, 
                        flierprops = dict(marker = "x", markersize = 3.5),
                        fliersize=1,
                        **boxplot_dct
                    )
                    ax.set_title(names[ax_number-1], fontsize=12*figscale)
                    ax.set_xlabel(f"{names[ax_number-1]}", fontsize=8*figscale)
                    ax.set_ylabel(f"attribute       ", fontsize=8*figscale, rotation=0)                
            
                elif plot_type=="probplot":
                    res = stats.probplot(df.iloc[:,ax_number-1], plot = ax, fit=True)
                    ax.set_title(
                        f"{names[ax_number-1]}\nR^2={np.round(res[1][2],3)}", 
                        fontsize=12*figscale)
                    ax.get_lines()[1].set_color('#F1480F')
                    ax.get_lines()[1].set_linewidth(3)      
                      
                      
                else:
                    pass
                    
                    
                # labels
                ax.xaxis.set_major_locator(plt.MaxNLocator(2))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))

                # remove axes, & ticks you dont want to see
                ax.yaxis.set_ticks_position("left")
                ax.xaxis.set_ticks_position("bottom")                
                
                # Remove ticks, and axes that you dot'n want, format the other ones,
                ax.spines["left"].set_visible(True)
                ax.spines["bottom"].set_visible(True)
                # ...
                ax.spines['top'].set_visible(False) # remove ...
                ax.spines['right'].set_visible(False) # remove ... 

                # Format ticks,
                ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
                ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
                ax.yaxis.set_ticks_position('left')# shows only that
                ax.xaxis.set_ticks_position('bottom')# shows only that                

    # Aestetics, 
    plt.tight_layout() # to avoid overlapping with the labels
    
    if title=="":
        pass
    else:
        plt.subplots_adjust(top=0.8)
    plt.show();
    
    
    
    
    
# Function, ..............................................................................
def feature_distribution(df, col):
    """
        plots
        1. histogram
        2. boxplot
        3. probanblity plot
        for one numerical variable in df
        - df; pandas dataframe
        - col; column name

        taken from: https://www.kaggle.com/mustafacicek/simple-eda-functions-for-data-analysis
    """
    
    skewness = np.round(df[col].skew(), 3)
    kurtosis = np.round(df[col].kurtosis(), 3)

    fig, axes = plt.subplots(1, 3, figsize = (21, 7))
    
    sns.kdeplot(data = df, x = col, fill = True, ax = axes[0], color = "#603F83", linewidth = 2)
    sns.boxplot(data = df, y = col, ax = axes[1], color = "#603F83",
                linewidth = 2, flierprops = dict(marker = "x", markersize = 3.5))
    stats.probplot(df[col], plot = axes[2])

    axes[0].set_title("Distribution \nSkewness: " + str(skewness) + "\nKurtosis: " + str(kurtosis))
    axes[1].set_title("Boxplot")
    axes[2].set_title("Probability Plot")
    fig.suptitle("For Feature:  " + col)
    
    for ax in axes:
        ax.set_facecolor("#C7D3D4FF")
        ax.grid(linewidth = 0.1)
    
    axes[2].get_lines()[0].set_markerfacecolor('#8157AE')
    axes[2].get_lines()[0].set_markeredgecolor('#603F83')
    axes[2].get_lines()[0].set_markeredgewidth(0.1)
    axes[2].get_lines()[1].set_color('#F1480F')
    axes[2].get_lines()[1].set_linewidth(3)
    
    sns.despine(top = True, right = True, left = True, bottom = True)
    plt.show()  
  
  
  
  
   
    
# Function, ..............................................................................  
def check_normality(df, names, p=0.05, verbose=False):
    """Check if the distribution is normal

    Parameters
    ----------
    data : vector of data to be tested
    show_flag : controls the display of data

    Returns
    -------
    python dataframe of p-values for different normality tests
   
    Coooments
    ---------
    Graphical test: if the data lie on a line, they are pretty much normally distributed
    """
    
    # work on copy of df subset
    assert isinstance(df, pd.core.frame.DataFrame), "df must be dataframe"
    df = pd.DataFrame(df.loc[:,names]).copy()    
    
    results = []
    for i, name in enumerate(names):
        
        # prepare
        data = df.loc[:,name]
        pVals = dict()

        # basic info
        pVals['attribute'] = name
        pVals['dtype'] = data.dtype
        pVals['data points'] = data.shape[0]
        pVals["na count"] = data.isnull().sum()

        # remove na
        data = data.dropna()
    
        # descriptive statistics
        pVals['skewness'] = np.round(data.skew(),4)
        pVals['kurtosis'] = np.round(data.kurtosis(),4)
        pVals['mean'] = np.round(data.values.mean(),4)
        pVals['sd'] = np.round(data.values.std(),4)
        
        # The scipy normaltest 
        ''' based on D-Agostino and Pearsons test that combines 
            skew and kurtosis to produce an omnibus test of normality.'''
        _, pVals['normaltest']    = stats.normaltest(data)

        # Shapiro-Wilk test
        _, pVals['Shapiro-Wilk']    = stats.shapiro(data)

        # Alternatively with original Kolmogorov-Smirnov test
        _, pVals['Kolmogorov-Smirnov']    = \
                stats.kstest((data-np.mean(data))/np.std(data,ddof=1), 'norm')

        results.append(pVals)

        # info
        if verbose==True:
            if data.shape[0]<300:
                test_used = 'Shapiro-Wilk'
            else:
                test_used = 'Kolmogorov-Smirnov'
            
            # normality_test_res
            if pVals[test_used]>p:
                res_to_print = f"data are normally distributed ({test_used}, p>{p}, sample size={data.shape[0]})"
            else:
                res_to_print = f"data distrib. signifficantly departed from normality ({test_used}, p<{p}, sample size={data.shape[0]})"
                            
            # printed message:
            print(f"{i}, {name}: {res_to_print}")
        else:
            pass
    
    
    return pd.DataFrame(results)
    