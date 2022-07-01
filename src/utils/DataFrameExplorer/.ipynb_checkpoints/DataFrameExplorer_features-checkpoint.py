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
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import random
import glob
import re
import os
import seaborn as sns

from matplotlib import cm, colors
from matplotlib.ticker import MaxNLocator
from pandas.api.types import is_numeric_dtype


# Function, .................................................................
def show_df_exaples(df, var_names, n=3 ):
    '''
        returns dtype and examples of requested variables in input df"
        df : pandas datafram
        var_names : list  
        n : int, number of examples provided with each variable
        
        CAUTION: .loc funciton, that I am using to subset dataframe, 
        do not accept, column names that are misisng in dataframe
        
        example
        >>> res_list = show_df_exaples(df=X_model, var_names=test_variables)  
        >>> pd.DataFrame(res_list)
        # chech if class_nr is correct      
    '''
    
    # test and work on copy
    assert type(df)==pd.core.frame.DataFrame, "df dtype error, shodul be pandas dataframe"
    df = df.copy()    
        
    # collect dtpye and exaples
    res_list =[]
    for var_name in var_names:
        try:
            # subset series
            s = df.loc[:,var_name]

            # count unique values
            counted_values = s.value_counts(ascending=False)  

            # basic info on that feature
            one_var_res_dct = {
                "name":var_name,
                "dtype": s.dtype,
                "class_nr":counted_values.shape[0],
                # ..
                "instances": s.shape[0],
                "na": s.isnull().sum()
                }

            # count unique values
            counted_values = s.value_counts(ascending=False)        

            # get n, most frequent variables, or less
            for i in range(n):
                if i+1 <= counted_values.shape[0]:
                    one_var_res_dct[f"eg{i+1}_value"]=counted_values.index.values.tolist()[i]
                    one_var_res_dct[f"eg{i+1}_counts"]=f"{counted_values.iloc[i]}; ({np.round(counted_values.iloc[i]/s.shape[0]*100,1)}%)"
                else:
                    one_var_res_dct[f"eg{i+1}_value"]=None
                    one_var_res_dct[f"eg{i+1}_counts"]=None
        
        except:
            # add info that a given feature was not found
            one_var_res_dct = {
                "name":var_name,
                "dtype": "NOT FOUND",
                "class_nr":None,
                # ..
                "instances": None,
                "na": None
                }         
            for i in range(n):
                one_var_res_dct[f"eg{i+1}_value"]=None
                one_var_res_dct[f"eg{i+1}_counts"]=None
                    
        # add to results list
        res_list.append(one_var_res_dct)            

    return pd.DataFrame(res_list)
    

# Function, .................................................................
def show_unlisted_variables(df, var_list, n=2, verbose=False):
    '''
        prints variables that were not defined in lists/dct in df, opr were not present in df, 
        provides examples of each variable with show_df_exaples() function, 
        parameters:
        - df : pandas dataframe
        - var_list : list or dict with list, that will be concateneated into one list
        - n :  how many exmaples of eahc variable to present,
        retuns:
        - defined_var_examples, undefined_var_examples, unknownw_var_examples: all dataframes
        
        example:
        >>> defined_var_examples, undefined_var_examples, unknownw_var_examples = show_unlisted_variables(
        ...        df = X_model,
        ...        var_list = {"level_indicator_variables":level_indicator_variables, 
        ...                    "numeric_variables": numeric_variables,
        ...                    "one_hot_encoded_variables": one_hot_encoded_variables
        ...                   },
        ...        verbose=True
        ...   )    


        
        
    '''
    
    # test and work on copy
    assert type(df)==pd.core.frame.DataFrame, "df dtype error, shodul be pandas dataframe"
    df = df.copy()
    
    # get var names from input df
    df_var_list = df.columns.values.tolist().copy()
    
    # ...
    
    # list with pre-defined variables & provided_var_names 
    if isinstance(var_list, list):
        defined_var_list = var_list.copy()
    else:
        defined_var_list =[]
        for k,v in var_list.items():
            defined_var_list.extend(v.copy())
    provided_var_names = defined_var_list.copy() # for reporting, and loop, no chnages
    
    # ...
    
    # find undefined variables in df
    unknownw_var_list  = [] # not found in df
    undefined_var_list = df_var_list.copy() # we will remove all defined viarblesd from that list
    
    for var_name in provided_var_names:
        search_res = (pd.Series(df_var_list)==var_name).sum() 
        if search_res == 0:
            unknownw_var_list.append(var_name)
            defined_var_list.remove(var_name)
        else:
            undefined_var_list.remove(var_name)
            
    # get examples from each group
    defined_var_examples = show_df_exaples(df=df, var_names=defined_var_list, n=n)
    undefined_var_examples = show_df_exaples(df=df, var_names=undefined_var_list, n=n)
    unknownw_var_examples = show_df_exaples(df=df, var_names=unknownw_var_list, n=n)
    
    # report 
    if verbose==True:
        print(f"input df var: {len(df_var_list)}")
        print(f"provided var names: {len(provided_var_names)}")
        print("-----------------------------------")
        print(f"defined var: {len(defined_var_list)} - (these variable names were found in df) ")
        print(f"undefined var: {len(undefined_var_list)} - (these variable names are in df, and were not specified in input list)")
        print(f"unknownw var: {len(unknownw_var_list)} - (these variable names were provided in input list, but were missing in df.columns)\n")
    else:
        pass
    
    return (defined_var_examples, undefined_var_examples, unknownw_var_examples)
  
  
  
  

  

# Function, ..............................................................................
def nice_boxplot(df, yvar, xvar, figsize=(10, 5), order=True, cmap="seismic", 
                 title=None, patch_width=1, labelsize=20, ticklabelsize=8, max_x_labels=None):
    '''
        creates boxlots, of one numeric variable (yvar), clustered with >=1 ordinal/indicator variables,
        + caulates correlation between median in each cluster, and the. response/target variable, 
        and. orders. automatically, subplots, starting from the variable combinaiton wiht the highest corr. coef.
        
        - df       pandas dataframe, with target variable (numeric), and indicator variables (numeric, text, object, or int)
        - yvar     str, colname, with responsse variable name in df
        - axvars   list[str,...], colanmes of indicator. variables in df, 
        - title    str, figure title,
        - patch_width width of pathech behind boxplots, for aestetics
        - labelsize int, fontsize for title, and y/xlabels 
        - ticklabelsiz int, fontsize for x labels, 
        - max_x_labels   None, or int, if int, it will be the max nr of equally spaced x-axis ticklabels, 
                         if None, all class names will be displayed on x-axis,   
    '''
    
    # data
    df_sub = pd.DataFrame(df).loc[:,[xvar,yvar]].copy()
    df_sub.reset_index(drop=True, inplace=True)
    
    # find medians of each class, and order the classes acordingly, 
    grp           = df_sub.groupby(by=xvar)

    # turn classes, into integers == means
    key_values_df = grp.median().loc[:,yvar]
    classes       = key_values_df.index.values.tolist()
    medians       = key_values_df.values.tolist()
    
    # replace class values with class medians
    num_xvar = pd.Series([0]*df.shape[0])
    for c, m in zip(classes,  medians):
        idx = df_sub.loc[:,xvar]==c  
        num_xvar.iloc[idx]=m
    num_xvar.reset_index(drop=True, inplace=True)  

    # calculate correlation 
    if is_numeric_dtype(df_sub.loc[:,xvar]):
        corr_value = df_sub.corr().iloc[0,1]
    else:
        corr_value = pd.concat([num_xvar,df_sub.loc[:,yvar]], axis=1).corr().iloc[0,1]    

    # ... set colors ...
    
    # select colors for boxes related to number of instances, 
    '''series:  index=xvar names, for each box, values=counts in df_sub'''
    xvar_counted = df_sub.loc[:,xvar].value_counts().sort_values() # series order from
    min_xvar_count = xvar_counted.min()
    max_xvar_count = xvar_counted.max()
    
    # select colors, 
    color_palette_for_boxes = plt.get_cmap(cmap)(
        np.linspace(0, 1, max_xvar_count-min_xvar_count+1))    
    
    
    # ... create box_colot_palette ...
    '''ie. dct, with "xvar_name":color for sns.boxplot'''
    
    # dict, I did that becaise i had some ptorblems before, 
    bcp_palette_dct = dict(zip(xvar_counted.index.values.tolist(), xvar_counted.values.tolist())) 
    bcp_palette_dct_source = bcp_palette_dct.copy()
    
    # replace conts with color values, corresponding to that value in color_palette_for_boxes
    for key, value in bcp_palette_dct_source.items():
        bcp_palette_dct[key] = color_palette_for_boxes[value-min_xvar_count]

        
    # ... find orer of boxes ...
    '''if order==True, boxes are ordered with the max median in y value, on the right
       else, sorted by values in xvar, eg growing number, or alphabetically,
    '''
    if order==False:
        box_order = pd.Series(classes).sort_values().values.tolist() 
    else:
        tempdf = pd.DataFrame({"class":classes,"median":medians})
        ordered_class_median_df = tempdf.sort_values(by="median")
        box_order = ordered_class_median_df.loc[:,"class"].values.tolist() 
        
    # ... main figure ...
    
    # Figure,
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, facecolor="white")
    ax = sns.boxplot(
        y=yvar, x=xvar, data=df_sub, ax=ax, order=box_order,
        palette=bcp_palette_dct, linewidth=0.2, fliersize=1)
    
    # add trendline,
    if order==False:
        ax.plot(np.arange(len(classes)), medians, ls="--", color="black")
    else:
        ax.plot(np.arange(len(classes)),
                ordered_class_median_df.loc[:,"median"].values, ls="--", color="black")
        
    # title
    if pd.isnull(title):
        fig.suptitle(f"{xvar}\ncorr={np.round(corr_value,3)}", fontsize=labelsize)    
    else:
        fig.suptitle(f"{title}\ncorr={np.round(corr_value,3)}", fontsize=labelsize)
    
    # ... format axes ...
    
    # axes labels
    ax.set_xlabel(xvar, fontsize=labelsize)
    ax.set_ylabel(f'{yvar}', fontsize=labelsize)
    ax.set_ylim(df_sub.loc[:,yvar].min(),df_sub.loc[:,yvar].max())
    
    # ... set max nr of ticks & labels on x/y axis ...
    
    # y-axis lebels number
    locator=MaxNLocator(prune='both', nbins=5)
    ax.yaxis.set_major_locator(locator)
    
    # x.axis label nr
    if pd.isnull(max_x_labels):
        pass
    else:
        locator=MaxNLocator(prune='both', nbins=max_x_labels)
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
    ax.tick_params(axis='x', rotation=70, labelsize=ticklabelsize)

    # ... grid ...
    ax.grid(color="lightgrey", lw=0.5, ls=":", axis="y")

    # ... colorpathes behind boxplot ...
    patch_width = patch_width   # ie. 1 = grey patch for 1 and 1 break
    patch_color = "lightgrey"
    pathces_starting_x = list(range(-1, len(classes)-1, patch_width*2))
    for i, sx in enumerate(pathces_starting_x):
        if sx+patch_width+0.5 <= len(classes)+0.5:
            rect = plt.Rectangle((sx+0.5, 0), patch_width, df_sub.loc[:,yvar].max(), color=patch_color, alpha=0.2, edgecolor=None)
        elif sx > len(classes)+0.5:
            break
        else:
            final_patch_width = len(classes)+0.5-sx
            rect = plt.Rectangle((sx+0.5, 0), final_patch_width, df_sub.loc[:,yvar].max(), color=patch_color, alpha=0.2, edgecolor=None)
        ax.add_patch(rect)        
        
    # ... colorbar ...
    norm = colors.Normalize(min_xvar_count, max_xvar_count)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, location="right", aspect=20, pad=0.01, shrink=0.5)
    cbar.set_label('# of instances in each box', rotation=270, fontsize=labelsize/2)
    cbar.outline.set_visible(False)
    
    # ... lyout ...
    plt.tight_layout()
    plt.subplots_adjust(top=.85)
    plt.show();
    
    