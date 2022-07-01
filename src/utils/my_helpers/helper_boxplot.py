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



    
# Function, ....................................................................
def colored_boxplots(df, yvar, xvars, sep='__', title=None, bt_dct=dict()):
    ''' Creates boxlots, of one numeric variable (yvar), grouped with at least one, or more variables in xvar,
        box color corresponsids to the number of examples used to create them, 
        additonally, it caulates correlation between median in each cluster, and the response/target variable, 
        plus, it orders boxplots, with ascending median
        
        parameters:
        . df; pandas Dataframe
        . yvar; str, feature name for y-axis
        . xvars; str, of list, with feature names used to create boxes
        . sep; if xvars is a list, it joins their name to be used as x-axis labels for each box
        . title; None, or string, 
        . bt_dct; dict, with params for  nice_boxplot function, 
        
        returns:
        . matplotlib plot, 
    '''
    # test input df, & work on df copy,
    assert type(df) == pd.DataFrame, "Incorrect obj type"
    df = df.copy()
    
    # prepare selection feature
    if isinstance(xvars, str): 
        df.loc[:,"groups"]  = df.loc[:,xvars]
    else:
        # remove rows wiht nan in xvar's,
        df = df.dropna(subset=xvars)

        # build feature with all xvar's concatenated,  
        def join_func(x): 
            return sep.join([str(i) for i in x])
        df.loc[:,"groups"] = df.loc[:,xvars].apply(join_func, axis=1)

    # create plot title
    if pd.isnull(title):
        if isinstance(xvars, str):
            title = xvars
        else:
            title = " - ".join(xvars)
    else:
        pass
        
    # create boxplot
    nice_boxplot( df=df, yvar=yvar, xvar="groups",title=title, **bt_dct)





# Function, ....................................................................
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