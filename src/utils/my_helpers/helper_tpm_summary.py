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



# Function, ......................................
def tpm_summary(df, n=None, plot=False, deg=2, hue=None):
    ''' creates fast summary for gene expression file, with tmp, 
        it helps me to set, or test applied thresholds, for filtering
        
        parameters
        . df; pandas Data Frame
        . n; number of randomly selected columns for creating plots, and summary table
        . plot; bool, if True, it shows the plot (see below)
        . deg; degree for polyft function, used for making trneline on mea_tpm~sd.tpm plot
        
        returns
        . pandas series wiht tmp statistics - stats descibed directly in pd.series, 
        . plot; if plot=True, with 2 subplots, showing % of genes/sample and sample/gene with tmp>0
    
        future developments
        . trendline: https://stackoverflow.com/questions/66040288/python-smoothing-2d-plot-trend-line
            here with plotly: https://plotly.com/python/linear-fits/
    '''
    # ensure, all tmp's are numeric
    arr = df.values.astype("float64")
    
    # hue vector
    if(hue is None):
        hue = [0]*arr.shape[0]
    else:
        pass
    hue_unique = pd.Series(hue).unique()
    
    # subset df columns, eg sample 1000 columns out of 10000
    if(n is None):
        arr = arr
        n = arr.shape[1]
    else:    
        idx = np.random.choice(list(range(df.shape[1])), size=n, replace=False)
        arr = arr[:,idx].copy()
        
    # create summary for each gene/column
    res=[]
    
    # .. one columns for each hue value
    for hue_value in hue_unique:
        arr_sub = arr[hue==hue_value,:] 
        res.append({
            "sample nr":arr_sub.shape[0],
            "gene nr":arr_sub.shape[1],
            "- VALUES -":"",
            "min tpm value recorded": np.round(arr_sub.min(),1),
            "median tpm value recorded": np.round(np.median(arr_sub),1),
            "mean tpm value recorded": np.round(arr_sub.mean(),1),
            "max tpm value recorded": np.round(arr_sub.max(),1),
            "- DETECTED IN -":"",
            "mean % of genes expressed per sample": f'{100-np.round(((arr_sub==0).sum(axis=1)/n*100).mean(),1)}%',
            "% of genes not expressed in any sample":f'{np.round(((arr_sub==0).sum(axis=0)==arr_sub.shape[0]).sum()/n*100,1)}%', # all are zero
            "% of genes expressed in at least 50%  of samples":f'{np.round(((arr_sub==0).sum(axis=0)<=arr_sub.shape[0]/2).sum()/n*100,1)}%', 
          # expressed in at least half of the samples
            "% of genes expressed in all samples":f'{np.round(((arr_sub==0).sum(axis=0)<=0).sum()/n*100,1)}%', # expressed in all samples
        })

    res_df = pd.DataFrame(res, columns=hue_unique) 
    return pd.DataFrame(res)



# Function, ......................................
def tpm_plots(df, n=None, deg=2, hue=None,title=None, color=None):
    ''' creates fast summary for gene expression file, with tmp, 
        it helps me to set, or test applied thresholds, for filtering
        
        parameters
        . df; pandas Data Frame
        . n; number of randomly selected columns for creating plots, and summary table
        . plot; bool, if True, it shows the plot (see below)
        . deg; degree for polyft function, used for making trneline on mea_tpm~sd.tpm plot
        
        returns
        . pandas series wiht tmp statistics - stats descibed directly in pd.series, 
        . plot; if plot=True, with 2 subplots, showing % of genes/sample and sample/gene with tmp>0
    
        future developments
        . trendline: https://stackoverflow.com/questions/66040288/python-smoothing-2d-plot-trend-line
            here with plotly: https://plotly.com/python/linear-fits/
    '''
    # ensure, all tmp's are numeric
    arr = df.values.astype("float64")
    
    # set colors
    if(color is None):
        colors = ["navy", "darkorange", "yellowgreen", "cyan", "gold"]
    else:
        colors = [color]*100
    
    # hue vector
    if(hue is None):
        hue = [0]*arr.shape[0]
    else:
        pass
    hue_unique = pd.Series(hue).unique()
    
    # subset df columns, eg sample 1000 columns out of 10000
    if(n is None):
        arr = arr
        n = arr.shape[1]
    else:    
        idx = np.random.choice(list(range(df.shape[1])), size=n, replace=False)
        arr = arr[:,idx].copy()
        
    # create summary for each gene/column
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(9,3), facecolor="white")
    if(title is not None):
        fig.suptitle(title)
    else:
        pass
    
    # create subplots on each dataset provided with different hue value
    for color, hue_value in zip(colors, hue_unique):
        arr_sub = arr[hue==hue_value,:] 

        # subplot 1. in how many samples, tmp is>0
        nonzero_tmp_per_gene = (1-(arr_sub==0).sum(axis=0)/arr.shape[0])*100
        axs[0].hist(nonzero_tmp_per_gene, alpha=0.5, color=color, label=hue_value, align='mid')
        axs[0].set_xlim(0,100)
        
        # subplot 2
        nonzero_tmp_per_sample = (1-(arr_sub==0).sum(axis=1)/n)*100
        axs[1].hist(nonzero_tmp_per_sample, alpha=0.5, color=color, label=hue_value)
        
        # .. data
        xdata = arr_sub.mean(axis=0)
        ydata = arr_sub.std(axis=0)
        axs[2].scatter(y=ydata, x=xdata, marker='h', alpha=0.5, color=color, label=hue_value)
        
        # ... trendline
        z = np.polyfit(xdata, ydata, deg)
        x_new = np.linspace(0, xdata.max(), 300)
        p = np.poly1d(z)
        axs[2].plot(x_new, p(x_new), color="black")        
        
    # aestetics
    axs[0].set_title("% of samples with detected\nexpression/transcripts")
    axs[0].set_ylabel("frequency")
    axs[0].set_xlabel("% of samples")
    axs[0].grid(lw=0.2)

    # subplot 2. 
    axs[1].set_title("% og genes expressed\n per sample")
    axs[1].set_ylabel("frequency")
    axs[1].set_xlabel("% of genes/sample")
    axs[1].grid(lw=0.2)

    # subplot 3. mean tpm ~ sd
    axs[2].set_title("standard deviation~the mean\nfor each gene")
    axs[2].set_ylabel("sd")
    axs[2].set_xlabel("mean")
    axs[2].grid(lw=0.2)
    
    # add legend
    if len(hue_unique)>1:
        axs[0].legend()
        axs[0].legend()
        axs[0].legend()
    else:
        pass

    sns.despine()
    fig.tight_layout()
    plt.show();
    
    
    
# Function, .............................................. 
def qc_for_selected_genes(tpm_data, target_var, gene_list=None, prefix="tpm"):
    ''' Function that provides additional information on genes evaluated with compare_gene_expression() function, 
        It looks into original data on each postion, and we will extract the follpowing:
            * number of samples wiht missing information in eahc group
            * mean, median, sd, of Tpm's in each group
        
        parametrs:
        . tpm_data; pd. data frame, with tpm data, either raw, or log
        . target_var; pd Series, with len()==tpm_data.shape[0], 
        . gene_list; list with genes, or index names in tpm_data, that will be used to subset the data, 
        . prefix; str, prefix added to column names, eg if you wish to create qc for different tables, and join them together
        
        Caution: 
        * in order to have the same positons selected, and samples withoitu outliers, use 
            * ttest_results df, returned by compare_gene_exprasion funciton - to provide table ,a dn gene names
            * y_tranf, ie target variable, with removed outliers,  
            * if not, target variable is not provided it will use all the data,  
            * df wiht tpm, either raw, or log transformed
    '''
  
  
    # data prep

    # test input df, & work on df copy,
    assert type(tpm_data) == pd.DataFrame, "tpm_data Incorrect obj type"
    assert type(gene_list) == list, "ttest_table Incorrect obj type"
    tpm_data, gene_list = tpm_data.copy(), gene_list.copy()

    # check target var
    if target_var is not None:
        assert type(target_var) == pd.Series, "target_var Incorrect obj type"
        target_var = target_var.copy()
    else:
        target_var = pd.Series(["all"]*tpm_data.shape[0])
    unique_target_var = target_var.unique().tolist()


    # data processing 
    '''provide summary for eacg gene, 
       for each group in target variable
    '''

    # . custom function
    def no_tpm(x):
        '''custom function to estimate data completnes per gene'''
        return 1-(np.sum(x==0)/len(x))

    # . column names for results
    column_names = ['mean', 'median', 'std', 'PercExpr']
    if prefix is not None:
        column_names = [f'{prefix}_{x}' for x in column_names]

    # . empty list for the results    
    df_res_column_names =[]

    # . create summary table
    for i, one_target_var in enumerate(unique_target_var): 

        # subset the data for target var.
        tpm_data_sub = tpm_data.loc[target_var==one_target_var,:]

        # calulate mean, median, sd, and tpm==0, per gene and store them in df
        df_res = pd.DataFrame(pd.concat([
                tpm_data_sub.apply(np.mean, axis=0),
                tpm_data_sub.apply(np.median, axis=0),
                tpm_data_sub.apply(np.std, axis=0),
                tpm_data_sub.apply(no_tpm, axis=0)
            ],axis=1), 
            index=tpm_data.columns
        )

        # add prefixed column names
        if len(unique_target_var)>1:
            df_res.columns = [f'{x}_{one_target_var}' for x in column_names]
        else:
            df_res.columns = column_names        
        
        # store all the results together
        if i==0: df_res_full = df_res
        else: df_res_full = pd.concat([df_res_full, df_res], axis=1)
        
    # select genes and return the table 
    if gene_list is not None:
        return df_res_full.loc[gene_list ,:]
    else:
        return df_res_full