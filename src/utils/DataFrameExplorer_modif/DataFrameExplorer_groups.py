# ********************************************************************************** #
#                                                                                    #
#   Project: Data Frame Explorer                                                     #                         
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

import matplotlib.gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from IPython.display import display
from PIL import Image, ImageDraw
from matplotlib import colors
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype
from matplotlib.font_manager import FontProperties




# Function, ............................................................................    
def compare_groups_with_df_summary_plot(*, 
    df,
    fname,
    top_n=3,
    fig_size=(8,6),
    groupSize=5,
    barPlot_dct=dict(),
    verbose=False                     
):
    ''' plots annotated, stacked barplot, divided into smaller groups, 
        data from input dataframe are divided with classes in one feature provided, or no feature is used, 
        one baplot is created for each class, in that feature
        Parameters:
        . df; pandas DataFrame, 
        . fname; feature name
        . top_n; int, number of the most frequent class shown on each bar,
        . groupsize; int, controls the number of bars are plotted closed to each other, for redability, 
        . barPlot_dct; optional args, for more info see help for   df_summary_plot() function, 
        . verbose; defualt False, if True prints info messages,
        Returns:
        . shows matplotlib figure
        Comments: 
        please modify the funcioton, to return fig. object if you wish to use it for reports, outside jupiter notebook env.
    '''
    
    
    # global settings - legacy code, 
    add_all_groups=True
    
    # find unique class labels in a feature
    'sorry for that expression, it was late and I had no internet, it shoudl be improved'
    unique_labels = df[fname].unique().tolist()
    unique_labels = pd.Series(unique_labels).loc[pd.Series(unique_labels).isnull()==False].tolist()

    # create the figure
    plt.style.use("classic")
    fig, axs = plt.subplots(nrows=1, ncols=len(unique_labels), figsize=fig_size, facecolor="white")

    # get 
    for i, class_name in enumerate(unique_labels):
        # get the data
        df_sub = df.loc[df[fname]==class_name,:]
        
        # get one axis
        if len(unique_labels)==1:
            ax = axs
        else:
            ax = axs[i]
            
        # remove barlbales from further axes
        if i==0:
            short_bar_names=None
        else:
            short_bar_names=""    
        
        # prepare data directly from dataframe
        data_examples, _, top_values_perc = summarize_df(  
                df = df_sub, 
                nr_of_examples_per_category = top_n,
                csv_file_name = None, 
                save_dir = None,
                verbose=verbose
            )

        # Pie chart with nr of col with different dtypes in data df,
        df_filter = pd.Series([True]*data_examples.shape[0])

        # add annotated barplot to axis
        ax = simplified_annotated_barplot(
                        ax = ax,
                        data_examples = data_examples, 
                        top_val_perc = top_values_perc, 
                        df_filter = df_filter, 
                        fig_size = fig_size,
                        group_size = groupSize,
                        short_bar_names=short_bar_names,
                        **barPlot_dct
                    )
        
        ax.set_title(f'{class_name}\n{df_sub.shape[0]} examples')
        
    fig.tight_layout()
    plt.show()

    







# Function, ......................... 
def simplified_annotated_barplot(*, 
    # input data
    ax, 
    data_examples, 
    top_val_perc,                                                                        
    df_filter, 
    
    # plot aestetics
    plot_title="", 
    fig_size=(12,12), 
    fontsize_scale=1,
    examples_fontsize_scale=1,
    group_size=5,
    barplot_cmap="tab10",
    cmap_from=0, 
    cmap_to=0.5,
    adjust_top=0.8,
    short_bar_names=None
):
    
    
    '''
        Generates bar plot used to get fast information on data 
        in different column in large df
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * data_examples     DataFrame with large Dataframe summary, 
                            generated with  summarize_data_and_give_examples()
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()
        
        .
        * df_filter         list, with True/False for each row in data_examples & top_val_perc 
                            if True, the row will be displayed on barplot
                            
        * fig_size          tuple, (row lenght, col lenght), in inches
        * font_size         int, size of all fonts used on the plot
        * group_size        int, how many rows will be diplayes as group on y axis on horizonal barplot.
                            groups are divided by space == to one bar.
                            
        Returns             matplotlib axis object
        _________________   _______________________________________________________________________________
        
        * plt.figure 
    '''
    
    # basic fontsize:
    font_size=8
    
    # helper,
    def stacked_barh_one_level(*, f_ax, bar_pos, top, bottom, colors, edgecolor, labels):
        f_ax.barh(bar_pos, top,left=bottom, color=colors, edgecolor=edgecolor, label=labels, linewidth=0.5, height=0.6)
        return f_ax    
    
    
    # ............................................................
    # Names and group filtering

    # group names,
    group_names       = list(data_examples.name.loc[df_filter])

    # data for plot,
    data_completness  = 100-np.array(data_examples.NaN_perc[df_filter]).flatten()
    tick_description  = data_examples.name[df_filter]
    top_values        = top_val_perc.values[df_filter, :]
    top_data_examples = data_examples.examples[df_filter]
    group_description = data_examples.summary[df_filter]

    # rescale top values,so they are part of non-missing data
    for i in range(top_values.shape[1]):
        v = top_values[:,i]
        top_values[:,i] = (v*data_completness)/100
    all_remaining_values = data_completness-top_values.sum(axis=1)

    # join the data in one array, I had some problems here, 
    data_for_plot = np.round(np.c_[(np.round(top_values,1), all_remaining_values)],1)

    

    # ............................................................
    # order the bars,

    # find order of the bars, based on data completness,
    bar_order = np.arange(data_completness.shape[0])

    # add spaces between everyx n-th bar, 
    add_spacers = True
    if add_spacers==True:
        # add spaces between everyx 5th bar, 
        space_between_groups = 1
        new_br = bar_order.copy().flatten()
        group_top  = []
        group_bottom = []

        for i, j in enumerate(sorted(list(bar_order))):

            if i==0: 
                add_to_list, counter = 0, 0
                group_bottom.append(j)

            if i>0 and counter<group_size: 
                counter +=1       

            if counter==group_size:
                group_bottom.append(j+add_to_list+1)
                counter=0
                add_to_list +=space_between_groups; 

            new_br[bar_order==j]=j+add_to_list

        group_top = [x+group_size-1 for x in group_bottom]    
        group_top[-1] = np.max(bar_order)+add_to_list
        bar_order = new_br.copy()
 
    # ............................................................
    # barplot parameters; this was just to help me in long function !
    numeric_data_for_plot = data_for_plot # np array, 
    top_data_examples     = top_data_examples
    bar_position          = bar_order + 1
    group_description     = group_description
    bar_related_fontsize  = font_size
    
    # bar_names, (ytick labels),
    if pd.isnull(short_bar_names):
        bar_names = group_names
    else:
        bar_names = [short_bar_names]*len(group_names)
        
    # ............................................................   
    # barplot,

    # Set style and colors,
    bar_colors = plt.get_cmap(barplot_cmap)(np.linspace(cmap_from, cmap_to, data_for_plot.shape[1])) # different nr of colors,
    edge_colors = bar_colors.copy()
    bar_colors[-1,0:3] = colors.to_rgb("lightgrey")
    edge_colors[-1,0:3] = colors.to_rgb("grey")

    # set axis limits
    plot_top_value = np.max(bar_position)+1
    ax.set_ylim(0,plot_top_value)
    ax.set_xlim(0,120)

    # add top values as % od data completness,
    counter =[]
    add_top_values=True
    if add_top_values==True:
        counter = 0
        for i in list(range(data_for_plot.shape[1]))[::-1]:
            if counter == 0:
                bar_start = [0]*data_for_plot.shape[0]    
                bar_end   = data_for_plot[:,i]
            else:
                bar_start = bar_start+bar_end           
                bar_end   = data_for_plot[:,i] # bar end is hot tall is an individual bar, segment, not top point on a graph
            counter+=1

            # plot level on stacked plot
            ax = stacked_barh_one_level(
                 f_ax=ax, 
                 bar_pos=bar_position, top=bar_end, bottom=bar_start, 
                 colors=bar_colors[i], edgecolor=bar_colors[i], labels="test",
                 )

    # Add ticks on y axis, and names for each bar,
    ax.set_yticks(bar_position)
    ax.set_yticklabels(bar_names, fontsize=bar_related_fontsize*fontsize_scale, color="black")
    ax.set_xticks([0, 25,50,75,100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=bar_related_fontsize*fontsize_scale, color="black")

    # Format ticks,
    ax.tick_params(axis='x', colors='black', direction='out', length=4, width=2) # tick only
    ax.tick_params(axis='y', colors='black', direction='out', length=4, width=2) # tick only    
    ax.yaxis.set_ticks_position('left')# shows only that
    ax.xaxis.set_ticks_position('bottom')# shows only that

    # Remove ticks, and axes that you dot'n want, format the other ones,
    ax.spines['top'].set_visible(False) # remove ...
    ax.spines['right'].set_visible(False) # remove ...  
    ax.spines['bottom'].set_linewidth(2) # x axis width
    ax.spines['bottom'].set_bounds(0,100) # Now the x axis do not go under the legend
    ax.spines['left'].set_linewidth(2) # y axis width 

    # Add vertical lines from grid,
    ax.xaxis.grid(color='grey', linestyle='--', linewidth=1) # horizontal lines

    # axes desciption
    ax.set_xlabel(f"% of no-nan")

    # add text with data completness above each bar,
    add_text_wiht_data_completness_above_each_bar=True
    if add_text_wiht_data_completness_above_each_bar==True:
        for i in range(numeric_data_for_plot.shape[0]):
            text_y_position = bar_position[i]-0.3
            text_x_position = numeric_data_for_plot.sum(axis=1).tolist()[i]+2

            # text,
            text_to_display = "".join([str(int(np.round(numeric_data_for_plot.sum(axis=1).tolist()[i],0))),"%"])
            t = ax.text(text_x_position, text_y_position,  text_to_display, fontsize=bar_related_fontsize*fontsize_scale, color="darkred")
            #t.set_bbox(dict(facecolor="white", alpha=0.3, edgecolor="white"))
    else: 
        pass

    return ax


   
# Function, .............................             
def summarize_df(*, df, nr_of_examples_per_category=3, csv_file_name=None, save_dir=None, verbose=False):
    """
        Summary table, with basic information on column in large dataframes,
        can be applied to dafarames of all sizes, Used to create summary plots
        
        IMPORTANT: this is one of my oldest function, that I am using a lot, 
                   I will soon update it to something better, but generating the same outputs, 
                   and more resiliant to unknownw datatypes, 
        
        Parameters/Input              
        _________________   _______________________________________________________________________________ 
        
        . Input .
        * df                DataFrame to summarize
        * nr_of_examples_per_category
                            how many, top/most frequent records shoudl 
                            be collected in each column and used as examples of data inputs form that column
                            NaN, are ignored, unless column has only NaN
        
        . Saving .          The fgunction return Dataframe, even if file name and path to save_dir are not available
                            In that case the file are not saved.
        * csv_file_name     .csv file name that will be used to save all three dataFrames create with that function
        * save_dir          path
        
        Returns             
        _________________   _______________________________________________________________________________
        
        * data_examples.    DataFrame, summary of df, with the follwing values for each column imn df
                            . name                       : column name in df, attribute name
                            . dtype.                 : {"nan", if ony NaN werer detected in df, "object", "numeric"}
                            . NaN_perc               : percentage of missing data (np.nan, pd.null) in df in a given attirbute
                            . summary                : shor informtaion on type and number of data we shoudl expectc:
                                                       if dtype == "numeric": return min, mean and max values
                                                       if dtype == "object" : return number of unique classes
                                                              or messsage "all nonnull values are unique"
                                                       if dtype == nan       : return "Missing data Only"                        
                            . examples                : str, with reqwuested number of most frequent value examples in a given category
                            . nr_of_unique_values         : numeric, scount of all unique values in a category 
                            . nr_of_non_null_values   : numeric, count of all non-null values in a category
        
        * top_val_perc      DataFrame with % of the top three or most frequence records in each column 
                            in large dataframe that was summarized with summarize_data_and_give_examples()

        * top_val_names     DataFrame, same as top_val_perc, but with values, saved as string
 
    """    
    
    
    assert type(df)==pd.DataFrame, "ERROR, df must be pandas dataframe"
    
    # info
    if csv_file_name!="none" and save_dir!="none":
        if verbose == True:
            print("\n! CAUTION ! csv_file_name shoudl be provided wihtout .csv file extension!")
        else:
            pass

    # create df,
    col_names   =["All_values_are_unique", "Nr_of_unique_values", "Nr_of_non_null_values", 
                  "Examples","dtype", "nr_of_all_rows_in_original_df"]
    df_examples = pd.DataFrame(np.zeros([df.shape[1],len(col_names)]), columns=col_names, dtype="object")

    # add category names
    df_examples["name"] = df.columns

    # add NaN percentage,
    nan_counts_per_category = df.isnull().sum(axis=0)
    my_data = pd.DataFrame(np.round((nan_counts_per_category/df.shape[0])*100, 5), dtype="float64")
    df_examples["NaN_perc"] = my_data.reset_index(drop=True)

    # add nr of no NaN values
    my_data = df.shape[0]-nan_counts_per_category
    df_examples["Nr_of_non_null_values"] = my_data.reset_index(drop=True)

    # add "nr_of_all_rows_in_original_df"
    df_examples["nr_of_all_rows_in_original_df"] = df.shape[0]
    
    # these arr will be filled for future bar plot
    arr_example_percentage = np.zeros([df_examples.shape[0],nr_of_examples_per_category])
    arr_example_values     = np.zeros([df_examples.shape[0],nr_of_examples_per_category],dtype="object")
    
    
    
    # add examples and counts
    for i, j in enumerate(list(df.columns)):

        # add general data  ..............................................

        # number of unique nonnull values in each column,
        df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = df.loc[:,j].dropna().unique().size

        
        #  internal function helpers .....................................

        # categorical data, fillin_All_values_are_unique
        def fillin_All_values_are_unique(*, df_examples, df):
            if (df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]==0).values[0]: 
                return "Missing data Only"
            elif ((df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]>0).values[0]) and (df.loc[:,j].dropna().unique().size==df.loc[:,j].dropna().shape[0]): 
                return "all nonnull values are unique"
            elif ((df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"]>0).values[0]) and (df.loc[:,j].dropna().unique().size!=df.loc[:,j].dropna().shape[0]): 
                return f"{int(df_examples.Nr_of_unique_values[df_examples.name==j].values[0])} classes"
            else:
                pass
            
            
            
        # fill other columns ..............................................

        # this is auto-fill in case there is no data
        if df[j].isnull().sum()==df.shape[0]:
            # (df_examples.loc[df_examples.name==j,"NaN_perc"]==100).values[0]: this value was rounded up/down and was returning false positives!!!!
            df_examples.loc[df_examples.name==j,"All_values_are_unique"] = "missing data only"
            df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"] = 0 # it should be 0, but i overwrite it just in case
            df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = 0 # it should be 0, but i overwrite it just in case
            df_examples.loc[df_examples.name==j,"Examples"] = "missing data only"
            df_examples.loc[df_examples.name==j,"dtype"] = "missing data only" # because I dont want to use that in any further reading

        # in other cases, we can create data examples, from nonnull values, depending on their type,
        else:

            if is_string_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "text"
                
                # All_values_are_unique,
                # use helper function, to find if there are only unique categorical values eg: url, place example in dct,  
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = fillin_All_values_are_unique(df_examples=df_examples, df=df)

                # Examples,
                count_noNa_values_sorted = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                perc_noNa_values_sorted  = count_noNa_values_sorted/np.sum(count_noNa_values_sorted)*100
                s        = perc_noNa_values_sorted.iloc[0:nr_of_examples_per_category].round(1)
                if len(s.index.values.tolist())>0:
                    ind      = pd.Series(s.index).values.tolist()
                else:
                    ind = [""]*nr_of_examples_per_category
                df_examples.loc[df_examples.name==j,"Examples"] = ";".join([str((x,y)) for x,y in zip(["".join([str(x),"%"]) for x in list(s)],ind)])

                            
                # add examples to arr for plot
                arr_example_percentage[df_examples.name==j,0:s.values.size]=s.values[0:nr_of_examples_per_category] 
                arr_example_values[df_examples.name==j,0:s.values.size]= np.array(s.index)[0:nr_of_examples_per_category]


            elif is_numeric_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "numeric"
                
                # All_values_are_unique,
                x = list(df.loc[:,j].dropna().describe()[["min", "mean", "max"]].round(3))
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = f"{round(x[0],2)} // {round(x[1],2)} // {round(x[2],2)}"

                # Examples,
                ordered_values = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                ordered_values = (ordered_values/df.loc[:,j].dropna().shape[0])*100
                df_examples.loc[df_examples.name==j,"Examples"] = ";".join([str((x,y)) for x,y in zip(
                    ["".join([str(int(np.ceil(x))),"%"]) for x in list(ordered_values)][0:nr_of_examples_per_category],
                    list(ordered_values.index.values.round(3))[0:nr_of_examples_per_category])])

                # add examples to arr for plot
                vn = np.array(ordered_values.index)[0:nr_of_examples_per_category]
                vp = ordered_values.values[0:nr_of_examples_per_category]
                arr_example_values[df_examples.name==j,0:ordered_values.size] = vn
                arr_example_percentage[df_examples.name==j,0:ordered_values.size] = vp
                
            elif is_datetime64_any_dtype(df[j]): 

                # dtype,
                df_examples.loc[df_examples.name==j,"dtype"]= "datetime"
                
                # variable summary,
                first_and_last_date = [str(x) for x in list(df.loc[:,j].dropna().describe()[["first", "last"]].dt.strftime('%b %d %Y'))]
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = f"{first_and_last_date[0]} - {first_and_last_date[1]}"

                # Examples,
                ordered_values = df.loc[:,j].dropna().value_counts().sort_values(ascending=False)
                ordered_values = (ordered_values/df.loc[:,j].dropna().shape[0])*100
                df_examples.loc[df_examples.name==j,"Examples"] =";".join([str((x,y)) for x,y in zip(
                    ["".join([str(np.round(x,3)),"%"]) for x in list(ordered_values)][0:2], 
                    list(pd.Series(ordered_values.index[0:2]).dt.strftime('%b-%d-%Y %H:%M').values))])# i add only two, because these are really long, 

                # add examples to arr for plot
                vn = list(ordered_values.index)[0:nr_of_examples_per_category]
                vp = ordered_values.values[0:nr_of_examples_per_category]
                arr_example_values[df_examples.name==j,0:ordered_values.size] = vn
                arr_example_percentage[df_examples.name==j,0:ordered_values.size] = vp     
                
            else:
                pass

                # (df_examples.loc[df_examples.name==j,"NaN_perc"]==100).values[0]: this value was rounded up/down and was returning false positives!!!!
                df_examples.loc[df_examples.name==j,"All_values_are_unique"] = "Uknown datatype"
                df_examples.loc[df_examples.name==j,"Nr_of_non_null_values"] = 0 # it should be 0, but i overwrite it just in case
                df_examples.loc[df_examples.name==j,"Nr_of_unique_values"] = 0 # it should be 0, but i overwrite it just in case
                df_examples.loc[df_examples.name==j,"Examples"] = "Uknown datatype"
                df_examples.loc[df_examples.name==j,"dtype"] = "Uknown datatype" # because I dont want to use that in any further reading


                
    # reorder column, so the look nicer, when displayed, 
    df_examples = df_examples.loc[:,["name", "dtype", 'NaN_perc', 'All_values_are_unique', 'Examples', 
                                     'Nr_of_unique_values', 'Nr_of_non_null_values', 'nr_of_all_rows_in_original_df']]   
    # rename some collumns, for PEP8 compliance,
    df_examples = df_examples.rename(columns={'All_values_are_unique':'summary', "Examples": "examples", 
                                              "Nr_of_unique_values": "nr_of_unique_values", "Nr_of_non_null_values":"nr_of_non_null_values"})
    

    # turn two additional elements into df's, 
    df_example_values = pd.DataFrame(arr_example_values, index=df_examples.name)
    df_example_percentage = pd.DataFrame(arr_example_percentage, index=df_examples.name)
    
    #### save the files 
    if csv_file_name!=None and save_dir!=None:
        try:
            os.chdir(save_dir)
            df_examples.to_csv("".join([csv_file_name,".csv"]), encoding='utf-8', index=False)
            df_example_values.to_csv("".join(["top_val_names_",csv_file_name,".csv"]), encoding='utf-8', index=False)
            df_example_percentage.to_csv("".join(["top_val_perc_",csv_file_name,".csv"]), encoding='utf-8', index=False)
            
            # info to display,and table example, 
            if verbose==True:
                print(f"""the file: {csv_file_name} was correctly saved \n in: {os.getcwd()} \n""") 
            else:
                pass

        except:
            if verbose==True:
                Error_message = "THE FILE WAS NOT SAVED, \n save_dir and/or csv_file_name were incorrect, or one of them was not provided"
                print(f"""{"".join(["."]*40)},\n ERROR,\n the file: {csv_file_name},\n {Error_message} \n in: {os.getcwd()} \n{"".join(["."]*40)}""")
            else:
                pass
                
    else:
        if verbose==True:
            Error_message = "THE FILE WAS NOT SAVED, \n save_dir and/or csv_file_name were not provided "
            print(f"""{"".join(["."]*40)},\n CAUTION,\n the file: {csv_file_name},\n {Error_message} \n in: {os.getcwd()} \n{"".join(["."]*40)}""")
        else:
            pass
    
    return df_examples, df_example_values, df_example_percentage