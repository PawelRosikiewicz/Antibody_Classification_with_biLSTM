# ********************************************************************************** #
#                                                                                    #                    
#   Author: Pawel Rosikiewicz                                                        #
#   Contact: prosikiewicz(a)gmail.com                                                #
#                                                                                    #
#   License: MIT License                                                             #
#   Copyright (C) 2022.06.25 Pawel Rosikiewicz                                       #
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
from sklearn.utils import shuffle



# Function, .............................................................  
def unique_aa_counts_hist(unique_aa_counts, label, class_size="na", figsize=(4,3)):
    ''' creates historgam with data provided 
        in unique_aa_counts (list, with integers), 
        for class, names in labels (str)
        
        Caution: this is modified fucntion for ml, pipelines, data preparaiton step 
    '''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.suptitle(f"Class:{label}\nwith {class_size} instances\n") 
    ax.hist(unique_aa_counts, bins=21, histtype="stepfilled",
             color="grey", edgecolor="grey", linewidth=5, alpha=0.5)
    ax.set_xlim(0, 21)
    ax.set_xlabel("Number of uniqe AA per position")
    ax.grid(lw=0.5, ls=":")
    sns.despine()
    fig.tight_layout()
    plt.show();