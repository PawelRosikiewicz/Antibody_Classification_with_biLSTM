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
import matplotlib.patches as mpatches

from src.utils.helper_boxplot import colored_boxplots

# Function, .......................................
def plot_colored_boxplot(df, n, title=None, figsize=(20,5)):
    ''' Creates boxlots ordered by ascending median, colored by the number of non-missing values used to creat each box,  
          -> helper funciton specificaly for data_genes dataset, wiht colnames==gene name, and rowidx=sample id
        
        parameters:
        . df; pandas Dataframe
        . n; number of column idxs, used for making boxplot
        . title; None, or string, 

        returns:
        . matplotlib plot, 
    '''

    # just range
    bt_data = df.iloc[:,0:n].transpose().copy(); 
    bt_data = bt_data.stack().reset_index(drop=False)
    bt_data.columns = ["gene", "sample", "TPM"]

    # set parameters for boxlot,
    bt_dct = dict(figsize=figsize, order=True, 
            cmap="coolwarm", patch_width=2, max_x_labels=None, labelsize=12)

    # create boxplot for each categorical variable + tagret variable
    colored_boxplots(
        df=bt_data, 
        yvar="TPM", 
        xvars="gene",
        bt_dct = bt_dct,
        title=title
    )

























