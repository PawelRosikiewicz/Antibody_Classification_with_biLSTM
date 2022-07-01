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
import scipy.stats as stats




# Function, ..............................................................................................
def compare_gene_expression(x, y, method="median"):
    ''' simple funciton that calulates gene expression foldchnage between two classes of samples
        and ttstudent test, then it data frame, with list of all genes, sorted with pvalue from ttest, 
        from the most, to the least signifficant,
        . x, y - dataframe and pd. series repsectively, with the data, and labels, 
        . method - str, "median", of "mean"
        ...............
        important: ttest shoudl be used only with relatively large number of replicates in each gorup, 
        eg >20, For more informaiton see these articles:
           https://academic.oup.com/ije/article/39/6/1597/736515
           https://doi.org/10.1016/j.csbj.2021.05.040
    '''
    # test input df, & work on df copy,
    assert type(x) == pd.DataFrame, "x_train Incorrect obj type"
    assert type(y) == pd.Series, "y_train Incorrect obj type"
    x, y = x.copy(), y.copy()
        
    # add "tiny" value to x, to avoid dividing by 0, or log(0)
    tval = 0
    
    # check how many unique goups are in the target variable
    y_unique = y.unique().tolist()
        
    # donw wast time if there is only one group
    '''done only to keep the rest of script run smottly'''
    if y_unique==1:
        ttest=[1]*x.shape[1] 
        Log2FC=[0]*x.shape[1]        
    else:
        # divide the set into two group
        ttest=[]
        Log2FC=[]
        for idx in range(x.shape[1]):
            one_row = x.iloc[:,idx].values
            a = one_row[y==0]
            b = one_row[y==1]

            # .. ttest
            ttest.append((stats.ttest_ind(a, b).pvalue))

            # Log2FC
            if method=="median": Log2FC.append((np.median(b)+tval)-(np.median(a)+tval))
            if method=="mean": Log2FC.append((np.mean(b)+tval)-(np.mean(a)+tval))

    # store results in nice dataframe
    results = pd.DataFrame([ttest,Log2FC]).transpose()
    results.columns = ['Pval', 'Log2FC']
    results.index = x.columns   
    
    # calculate LogPval
    results['LogPval'] = -(np.log(results.loc[:,"Pval"]))
    return results.sort_values(by="Pval", ascending=True)



# Function, ..............................................................................................
def select_genes_and_make_volcano_plot(
    df, xname="Log2FC", yname="LogPval", pname="Pval", 
    title="Volcano Plot", figsize=(10,4),Ptr=0.05, Log2FCtr=2, create_plot=True
):
    ''' This fucntions, allows selecting differencially expressed 
        genes with PValue, and Log2 fold chnage thresholds, 
        it returns, the table wiht selected genes, (gene names are indexes),
        and if create_plot is True, it also returns volcano polot wiht selected genes, 
        and basic informaiton on their number
    
        parameters:
        . df; Pandas datagrame, input data
        . xname; def. "Log2FC"; column name wiht log 2 fold chnage data in df, also y-axis on a plot
        . yname; def."LogPval"; column name wiht -log10(p-value) data in df, also y-axis on a plot
        . pname; def."Pval"; column name wiht p-value data in df, used for adding colors, 
            on a scatter points, and selection fo the points for the table
        . title; str, added on top of the title on a volcano plot,
        . figsize, tuple, with two int,
        . Ptr; float, threshold on p-value 
        . Log2FCtr; float or int, threshold on Log2FC 
        . create_plot; if True, returns volcano plot       

    '''
    # test input df, & work on df copy,
    assert type(df) == pd.DataFrame, "x_train Incorrect obj type"
    df = df.copy()   
    

    # (A) prepare the data
    #.select upregulated & downregulatzed genes
    downregulated_genes = df.loc[(df.loc[:,xname]<(-Log2FCtr)) & (df.loc[:,pname]<=Ptr), :]
    upregulated_genes = df.loc[(df.loc[:,xname]>Log2FCtr) & (df.loc[:,pname]<=Ptr), :]    
    
    
    # (B) volcano plot
    if create_plot==True:

        # make figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        # scatter plot - all points, 
        ax.scatter( x=df.Log2FC, y=df.LogPval, 
                    marker='o', alpha=1, s=6, color="grey")

        # add  subpltitle  with info on pre-selected postions
        cut_offs = f'cut-off of Pval={Ptr}, and Log2FC=±{Log2FCtr}' 
        position_nr = f'DOWNREGULATED: {downregulated_genes.shape[0]}, UPREGULATED:{upregulated_genes.shape[0]}'
        fig.suptitle(f'{title}\n{cut_offs}\n{position_nr}')       

        # upregulated and downregulated points wiht red and blue colors
        ax.scatter(
            x=downregulated_genes.loc[:,xname], 
            y=downregulated_genes.loc[:,yname], 
                    marker='h', alpha=1, s=5, color="blue")    
        ax.scatter(
            x=upregulated_genes.loc[:,xname], 
            y=upregulated_genes.loc[:,yname], 
                    marker='h', alpha=1, s=5, color="red")   

        # set limits, to make the plot more easy to real
        '''try, because sometimes you will find no up/donw regulated genes'''
        try:
            xlimits = np.abs([downregulated_genes.loc[:,xname].min(),upregulated_genes.loc[:,xname].max()])
            ax.set_xlim(-xlimits.max(), xlimits.max())    
        except:
            pass
          
        # horizontal and vertical lines with Pval and Log2FC thresholds
        ax.axhline(-(np.log(Ptr)), lw=0.5, ls="--", color="black")
        ax.axvline(-(Log2FCtr), lw=0.5, ls="--", color="black")
        ax.axvline(Log2FCtr, lw=0.5, ls="--", color="black")

        # ax aestetics
        ax.grid(lw=0.3)
        ax.set_xlabel("Log2FC")
        ax.set_ylabel("-Log10(Pval)")

        # fig layout
        fig.subplots_adjust(top=0.7)
        sns.despine()    
        plt.show();
    
    else:
        pass
    
    # (C) return table
    selected_genes = pd.concat([downregulated_genes, upregulated_genes], axis=0)
    return selected_genes




# Function, ..............................................................................................
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


