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


# Function, ............................................................................
def find_pattern(series, pat, verbose=False):
    '''
        I used that function when i don't remeber full name of a given column
        - series: list or pd. Series
        - pattern: str, or regex opattern used with pandas string funcitons, str literal, 
    '''
    assert isinstance(series, list) or isinstance(series, pd.core.series.Series), "imput var: series must be list, or pandas series"
    series = pd.Series(series)
    res = series.loc[series.str.contains(pat)].values.tolist()
    
    if verbose==True:
        if len(res)==0:
            print("No match") 
        else:
            print(res)

    return res
  
  
  
  
  
  
# Function, ...........................................................................................
def remove_nan_and_duplicates(*, df, col_threshold="none",  row_threshold="none", drop_duplicates=True, verbose=False):
    
    """ 
        =================   ===============================================================================
        Property            Description
        =================   ===============================================================================
        
        * Function          Removes column and rows with certain amount of missing data, in that order.
                            ie. first it removes columns, then rows, 
                            finally, it removes complete row duplicates, with 
                            pandas.DataFrame.drop_duplicates(keep='first')
                            
                            Thresholds examples & intepretation:
                            
                            100 - only row/col wiht no missing data will be accepted
                            95  - col/rows with 5% or more missing data will be removed
                            5   - col/row with up to 95% of missing data will be accepted
                            0   - even row/cols containsing only missing data will pass
                            
        Parameters/Input              
        _________________   _______________________________________________________________________________  

        * df                Pandas Dataframe
        * col_threshold     "all","none" or int between 0 and 100
        * row_threshold     -||- see col_threshold 
        * drop_duplicates   bool, True, by default,  

        Returns              
        _________________   _______________________________________________________________________________  

        * DataFrame         by Pandas, with removed dimensions,
        * display messages. on chnages in Dataframe dimensions,
    """

    # collect info, ...................................................     
    df_shape_before = df.shape
   
    # Missing data ................................................

    # .. First, remove columns,
    if col_threshold=="all":
        df = df.dropna(how=col_threshold, axis=1) # removes columns with NaN in all columns 
    elif col_threshold=="none": 
        "do nothing"
    else:
        Tr = int(np.ceil(df.shape[0]*(col_threshold)/100))
        df = df.dropna(thresh=Tr, axis=1) # keep, columns with Tr number of No-NaN     
             
    # .. Second, remove rows,   
    if row_threshold=="all":        
        df = df.dropna(how=row_threshold, axis=0) # removes rows with NaN in all columns 
    elif row_threshold=="none": 
        "do nothing"
    else:
        Tr = int(np.ceil(df.shape[1]*(row_threshold)/100))
        df = df.dropna(thresh=Tr, axis=0) # keep, rows with Tr number of No-NaN   

    # .. collect info,  
    df_shape_without_nan = df.shape
  


    # Remove Complete row duplicates ................................................

    # .. keep only first complete duplicated row,
    if drop_duplicates==True:    
        df = df.drop_duplicates(keep='first')
    
    # .. for info,
    df_shape_without_duplicates = df.shape    
    
    # diplay messages with info, ...................................................
    if verbose==True:
        print(f"""\n{"".join(["-"]*80)} \n Removing NaN and Duplicates \n{"".join(["-"]*80)}""")
        print(f"""
                Removed col's with less then {col_threshold}% of non-missing data 
                Removed rows  with less then {row_threshold}% of non-missing data 
                Removed completely duplicated rows : {drop_duplicates}
                ...
                df.shape before:        {df_shape_before}, 
                df.shape withhout nan:  {df_shape_without_nan},
                df.shape withhout dupl: {df_shape_without_duplicates},""", end="\n\n")
        print(f"""CAUTION ! Missing data are often clustered, thus removing rows and columns 
              at the samne time may drasticaly reduce the amount of data in returned df
              and result in different, then intended values of NaN/row and NaN/column""")
    
    if verbose==False:
        pass
        
    # return ----------------------------------------------------------------- end 
    return df.copy()







