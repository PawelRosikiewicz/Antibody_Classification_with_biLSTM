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

import os
import sys
import re # module to use regular expressions, 
import glob # lists names in folders that match Unix shell patterns
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn import set_config
from sklearn.preprocessing import RobustScaler # creates custom transfomers
from sklearn.preprocessing import FunctionTransformer # creates custom transfomers
from sklearn.pipeline import make_pipeline, Pipeline # like pipeline function, but give step names automatically, 
from sklearn.compose import ColumnTransformer # allows using different transformers to different columns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer # skleanr transformers,
from sklearn.decomposition import PCA

# helper function for qc and piepline calibration
from src.utils.helper_tpm_summary import tpm_summary
from src.utils.helper_tpm_summary import tpm_plots
from src.utils.helper_cluster_histogram import spearman_clustermap
from src.utils.helper_boxplot import colored_boxplots
from src.utils.helper_colored_boxplot import plot_colored_boxplot
from src.utils.helper_gene_expression_clustermap import gene_expression_clustermap
from src.utils.helper_tpm_histogram import tpm_hist
from src.utils.helper_tpm_hist_per_gene import tpm_hist_per_gene
from src.utils.helper_PCA_plots import pca_plot_and_scree_plot
from src.utils.helper_PCA_plots import tsne_plot

# my custom transformers & gene seleciton tools
from src.utils.preprocessing_spearman_filter import SpearmanFilter # to remove sample outliers, 
from src.utils.preprocessing_zero_value_filter import ZeroValueFilter # to remove genes with no tpm in most of the samples
from src.utils.preprocessing_gene_selection_tools import compare_gene_expression 
from src.utils.preprocessing_gene_selection_tools import qc_for_selected_genes
from src.utils.preprocessing_gene_selection_tools import select_genes_and_make_volcano_plot



# -------------------------------------------------------------------------------------
# SECTION A. CUSTOM TRANSFORMERS
# -------------------------------------------------------------------------------------


# Function, .................................
def log_transformer(x):
    ''' log1p scaller for input dataframe (x)
        CAUTION: log1p returns runtime warning if negative data are used'''
    x = x.copy()
    x = x+1
    log_tr = make_pipeline( 
        FunctionTransformer(np.abs), # or else, you may get some errors, 
        FunctionTransformer(np.log2, validate=False),
    )
    x_log = pd.DataFrame(
        log_tr.fit_transform(x),
        columns=x.columns
    )
    return x_log


# Function, .................................
def rebuild_df(arr, dfdonor):
    '''takes an array, and tunrs into df, 
        with idx, and colnales form the donor
        . arr; numpy arr, 
        . df; pandas dataframe providing col/row names/indexes,
    '''
    df = pd.DataFrame(arr, columns=dfdonor.columns, index=dfdonor.index)
    return df


# Function, .................................
def rebuild_transposed_df(Tarr, dfdonor):
    '''takes an array, and tunrs into df, 
        with idx, and colnales form the donor
        . arr; numpy arr, 
        . df; pandas dataframe providing col/row names/indexes,
    '''
    df = pd.DataFrame(Tarr.T, columns=dfdonor.columns, index=dfdonor.index)
    return df




# -------------------------------------------------------------------------------------
# SECTION B. HELPER FUNCITONS FOR QC AND VERBOSE REPORTS
# -------------------------------------------------------------------------------------


# Function, ..............................................................................................
def make_tpm_summary(df_list, name_list=None):
    ''' PIPELINE HELPER FUNCTION
        Creates tpm cummary table for several dataframes provided in the list (df_list)
        colnames - list with columns names for new summary table 
        for more ifo see help for tpm_summary() 
    '''
    for i, df in enumerate(df_list):
        if i==0:
            summary_table = tpm_summary(df=df)
        else:
            summary_table =  pd.concat([summary_table, tpm_summary(df=df)])
    
    # transpose for nice looking output
    summary_table = summary_table.transpose() 
    
    # add column names corresponding to different df's
    if(name_list is not None):
        summary_table.columns=name_list
    else:
        summary_table.columns=list(range(len(df_list)))
    
    return summary_table
      
    
# Function, ..............................................................................................
def verbose_info(dtype, global_qc_report, x_samples_accepted, x_samples_removed, ttest_results, x_transf_deg, verbose=0):
    ''' PIPELINE HELPER FUNCTION
        small helper function to print-out messages for verbose 1 and 2
        for object description, please see DATA PREPROCESSING PIPLINE
    '''
    if verbose==1 or verbose==2:
        print(f'\n{"".join(["."]*60)}\n {dtype} - Step 1-5\n{"".join(["."]*60)}\n')
        display(global_qc_report)
            
        if verbose==2:
            print(f'\n OUTLIER REMOVAL')
            print(" ---- ACCEPTED", x_samples_accepted )
            print(" ---- REMOVED", x_samples_removed )     
            print(f'\n T-TEST RESULTS WIHT  TOP 10 GENES')                
            display(ttest_results.head(10))
            print(f'\n TRANFORMED INPUT DATA 10 GENES') 
            display(x_transf_deg.head(10))
        else:
            pass
    
    else:
        pass
    
    
# Function, ..............................................................................................
def plot_heatmaps_and_PCA_plots(deg_table, x_transf, y_transf, data_genes_log):
    ''' PIPELINE HELPER FUNCTION
        generates heatmasp with spearman_clustermap and gene_expression_clustermap funcitons,
        and pca plots on data prepared wiht te pipeline, 
        
        parameters
        . deg_table; pd. data frame with DE genes, selected by the pipeline
        . x_transf; pd. data frame with data from pipeline after step 4   
        . y_transf; pd. series frame with target variable, after step 4
        . data_genes_log; pd. data frame with log2p data
    '''
    
    # to silence error on depreciated function, 
    warnings.filterwarnings("ignore")
    
    a=True # legacy, from trials, 
    if a == True:
        # get gene names
        top_scores = deg_table.shape[0]
        gs_top =  deg_table.sort_values(by="Pval", ascending=False).index[0:top_scores]

        # crate spearma clustermap to see if these gene allows better separation between target classes
        title = f"SIMILARITY BEWEEN SAMPLES\nBASED ON GENE EXPRESSION IN THE TOP {top_scores} \ndifferencially expressed genes"
        print(f"{''.join(['.']*60)}\n{title}\n{''.join(['.']*60)}")
        
        # plot
        spearman_clustermap(
            df=x_transf.loc[:, gs_top.values.tolist()].transpose(), 
            labels=y_transf,  
            n=None, 
            figsize=(7,7),
            title=f"SAMPLE SIMILARITY - top {top_scores}"
        )

        # check if the selected gene "really" have different expression
        title = f"GENE EXPRESSION IN THE TOP \n {top_scores} differencially expressed genes"
        print(f"{''.join(['.']*60)}\n{title}\n{''.join(['.']*60)}")
        
        # plot
        gene_expression_clustermap(
            df=data_genes_log.loc[:, gs_top.values.tolist()], 
            labels=y_transf,  
            n=None, 
            figsize=(7,7),
            title=f"GENE EXPRESSION - top {top_scores}",
            cmap='coolwarm'
        )

        # get gene names
        title = f"PCA and tSNE analysis"
        print(f"{''.join(['.']*60)}\n{title}\n{''.join(['.']*60)}")
        
        # plots
        pca_plot_and_scree_plot(x_transf.loc[:, gs_top.values.tolist()], y_transf, scale="y")
    
    else:
        print("problem wiht heatmaps")
        
        
# Function, ..............................................................................................        
def create_sample_qc_table(sample_ID, x_samples_corr, cov_table):
    ''' PIPELINE HELPER FUNCTION
        generetas summary with ampout of missing data, and corr, at each sample in covariance data, 
        make sure that you removed outliers in x_corr results, for train data
        . x_samples_corr, x_log_filtered, cov_table; data frames, their names 
          correspond to objects used in  DATA PREPROCESSING PIPLINE
    '''
    # copy
    ct = cov_table.copy()
    x_corr = x_samples_corr.copy()
        
    # build table
    res = pd.concat([
        sample_ID,
        ct.isnull().sum(axis=1)/ct.shape[0],
        x_corr
    ], axis=1)
        
    # add column names & return
    res.columns=['sample_ID', 'no_tpm_prop', "spearman_corr"]
    return res



# -------------------------------------------------------------------------------------
# SECTION C. DATA PREPROCESSING PIPELINE
# -------------------------------------------------------------------------------------

# Function, .............................................................................................. 
def data_preprocessing_pipeline(
    x_train, y_train, cov, cov_data_preprocessor, x_test_list=None, y_test_list=None, cov_list=None, 
    zv_filter_dct=dict(), sp_filter_dct=dict(), select_genes_dct=dict(), 
    verbose=0, pca_n_components=0, make_hist=False):
    ''' This my full preprocessing pipeline, that is fully described in notebook introduction, 
        in short, it formats tpm data, removes genes with too much missing inf (ie. tmp==0), 
        removes outliers, with spearman filter, identifies differentialy expressed genes, perfomrs PCA, 
        and reports results wiht plots, histograms and tables, that i created in notebook 02

        parameters:
        . x_train, pd.DataFrame with tpm data
        . y_train, pd Series, with target data
        . cov, pd.dataframe, with covariants data
        . cov_data_preprocessor, skleanr preprocessor for cov.
        . ...
        . x_test_list, list with pd.DataFrame with tpm data tables, for test and validation datasets
        . y_test_list=None, -||- with target data for test/validation, None, if not used
        . cov_list=None, , -||- with cov tables for test and validation 
        . ...
        . zv_filter_dct=dict(), parameters, for ZeroValueFilter() # custom build for the project 
        . sp_filter_dct=dict(), -||- for SpearmanFilter() # custom build for the project 
        . select_genes_dct=dict(), -|| for DE genes selection, on Pval, and Log2FC
        . pca_n_components=0, if 0, no PCA, if None, or int, it performs pca
        . make_hist; if True, it provides histograms, as in the notebook 02, for selected genes, 
        . ...
        . verbose, if 1, it provides basic info, if 2 it provides additional tables 
            form selection steps, for debugging
        
        returns:
        . data dict with transfomed x_tains, y_train, and cos table, each in dict for train, and test0-n
        . qc dict, with reports on global qc selection process, gene and sample qc, in the final datasets, 
        + plots mentioned in abstract

    '''

    # - instanciate transformers ........................................... 
    robust_scaler = RobustScaler() 
    #robust_scaler =  QuantileTransformer(output_distribution='normal')
    zv_filter = ZeroValueFilter() # custom build for the project 
    sp_filter = SpearmanFilter() # custom build for the project 
    
    # - instanciate pca - optional
    if pca_n_components is None: 
        pca = PCA(n_components=None)
        use_pca = True
    else:
        if pca_n_components>0: 
            pca = PCA(n_components=pca_n_components)
            use_pca = True
        if pca_n_components==0: 
            use_pca = False
    
    # create lists for qc data and the datasets
    tpm_data = dict()
    target_data = dict()
    covariants_data = dict()
    
    # qc
    global_qc = dict()
    sample_qc = dict()
    gene_qc = dict()
    
    # verbose options
    if verbose==1 or verbose==2:
        select_genes_dct['create_plot']=True
        select_genes_dct['title']= f"train data"
    else:
        select_genes_dct['create_plot']=False
    
    
    # - process train data ...........................................

    # setup
    
    # . work on df copy,
    x, y =  x_train.copy(), y_train.copy()  
    cov_table =  cov.copy()  
    
    
    # pipeline - part 1 - data cleaning
    
    # . step 1. log1p to combat heteroscedascity,
    x_log = log_transformer(x)
    x_log = rebuild_df(x_log, x)
    # ... reset index 
    x_log.reset_index(drop=True, inplace=True)
    
    # . step 2. remove genes with too much na and noise, 
    x_log_filtered = zv_filter.fit_transform(x_log, **zv_filter_dct)

    # . step 3. apply  to results from each sample
    x_log_filtered_scaled = robust_scaler.fit_transform(x_log_filtered)
    x_log_filtered_scaled = rebuild_df(x_log_filtered_scaled, x_log_filtered)    
    # ... reset index 
    x_log_filtered_scaled.reset_index(drop=True, inplace=True)
    
    # . step 4. remove potential outliers from train data
    x_transf, y_transf = sp_filter.fit_transform(
        x=x_log_filtered_scaled, 
        y=y, 
        **sp_filter_dct
        )
    x_samples_removed = sp_filter._train_samples_removed
    x_samples_accepted = sp_filter._train_samples_accepted
    x_samples_corr = sp_filter._train_samples_corr
    
    # ... reset indexes
    x_transf.reset_index(drop=True, inplace=True)  
    y_transf.reset_index(drop=True, inplace=True)  

    # ... remove outliers from covarinats table
    cov_table = cov_table.iloc[x_samples_accepted, :]
    cov_table.reset_index(drop=True, inplace=True)        
        
      
    # pipeline - part 2 - DE Gene Selection
    
    # . prepare log(tmp) file for DE analysis (not, x_tranf !)
    de_x = x_log_filtered.iloc[x_samples_accepted,:]
    de_x.reset_index(drop=True, inplace=True)  
    
    # . find diff. expressed genes & store qc data
    ttest_results = compare_gene_expression(x=de_x, y=y_transf, method="mean")
    ttest_results_qc = qc_for_selected_genes(
        tpm_data=x_log, 
        target_var=y, 
        gene_list=ttest_results.index.values.tolist(), 
        )
    ttest_results = pd.concat([ttest_results, ttest_results_qc], axis=1)
    deg_table = select_genes_and_make_volcano_plot(
      ttest_results, **select_genes_dct) # potential breaking point - if no genes meet criteria

    # . subset the data in x_tranf and ()
    x_transf_deg =  x_transf.loc[:,deg_table.index.values.tolist()]

    
    # pipeline - part 3A - PCA
    if use_pca==True:
        pca.fit(x_transf_deg)
        x_components = pca.transform(x_transf_deg)   
        x_components = pd.DataFrame(x_components, index=x_transf_deg.index)
    else:
        x_components = x_transf_deg
  
    # pipeline - part 3B - covarinat data preprocessor
    cov_table_transf = cov_data_preprocessor.fit_transform(cov_table)

    # pipeline - part 4 - store the data
    tpm_data["train"] = x_components
    target_data["train"] = y_transf 
    covariants_data["train"] = cov_table_transf
    
    
    # pipeline - part 5 QC & info

    # . global_qc
    global_qc_report = make_tpm_summary(
        df_list=[x, x_log, x_log_filtered,  x_log_filtered_scaled, x_transf, x_transf_deg, x_components],
        name_list=['input', 'log', 'log_filtered',  'log_filtered_scaled', 
                   'outliers_removed', 'top_genes', "after pca"]
        )     
    sample_qc_report = create_sample_qc_table(
        sample_ID = pd.Series(x_samples_accepted),
        x_samples_corr = x_samples_corr.iloc[x_samples_accepted], 
        cov_table = cov_table
    )

    # . collect data qc
    global_qc[f"train"]=global_qc_report
    sample_qc[f"train"]=sample_qc_report
    gene_qc[f"train"]=deg_table    # i need to do second version for one table  

    # . verbose
    verbose_info("train", global_qc_report, x_samples_accepted, x_samples_removed, ttest_results, x_transf_deg, verbose=verbose)
    
    
    # . create hist, that will help you visualize whether selected genes are good, 
    if make_hist==True:
        plot_heatmaps_and_PCA_plots(
            deg_table = deg_table, 
            x_transf = x_transf, 
            y_transf = y_transf, 
            data_genes_log = x_log
        )
    else:
        pass

    # - process test & validation data ...........................................
    for i, x_te, y_te, cov_table in zip(list(range(len(x_test_list))), x_test_list, y_test_list, cov_list):

        # setup
        
        # . work on df copy,
        x, cov_table =  x_te.copy(), cov_table.copy()   
        
        # . test data not always have target variable 
        if y_te is None: y=pd.Series(["uknownw"]*x_te.shape[0])
        else: y=y_te.copy()
        
        # define verbose options
        if verbose==1 or verbose==2: 
            select_genes_dct['title']= f"test data {i}"
        else:
            pass
    
        # pipeline - part 1 - data cleaning

        # . step 1. log1p to combat heteroscedascity,
        x_log = log_transformer(x)
        x_log = rebuild_df(x_log, x)

        # . step 2. remove genes with too much na and noise, ---------
        x_log_filtered = zv_filter.transform(x_log)

        # . step 3. apply  to results from each sample
        x_log_filtered_scaled = robust_scaler.transform(x_log_filtered)
        x_log_filtered_scaled = rebuild_df(x_log_filtered_scaled, x_log_filtered)
        
        # . step 4. remove potential outliers from train data
        '''IN TRANSFORM, WE DO NOT REMOVE OUTLIERS, BUTT WE CHECK THEIR QUALITY'''
        x_transf, y_transf = x_log_filtered_scaled, y
        _ , _ = sp_filter.transform(
            x=x_log_filtered_scaled, 
            y=pd.Series(y), 
            inform = True
            )
        x_samples_removed = sp_filter._test_samples_removed
        x_samples_accepted = sp_filter._test_samples_accepted
        x_samples_corr = sp_filter._test_samples_corr

        # pipeline - part 2
        '''in test samples we are using log files directly, no samples are removed'''
        
        # . select DE genes, based on train data, 
        """subset the data in x_tranf with pre-selected genes from train data"""
        genes_to_subset = gene_qc["train"].index.values.tolist()
        x_transf_deg =  x_transf.loc[:,genes_to_subset]
        
        
        # pipeline - part 3A - PCA
        if use_pca==True:
            x_components = pca.transform(x_transf_deg)   
            x_components = pd.DataFrame(x_components, index=x_transf_deg.index)
        else:
            x_components = x_transf_deg       

        # pipeline - part 3B - covarinat data preprocessor
        cov_table_transf = cov_data_preprocessor.transform(cov_table)

    
        # pipeline - part 4 - store the data
        tpm_data[f"test{i}"] = x_components
        target_data[f"test{i}"] = y_transf 
        covariants_data[f"test{i}"] = cov_table_transf # no modificaiton in test data   

        
        # pipeline - part 5 - QC
        
        # . global_qc
        global_qc_report = make_tpm_summary(
            df_list=[x, x_log, x_log_filtered,  x_log_filtered_scaled, x_transf, x_transf_deg, x_components],
            name_list=['input', 'log', 'log_filtered',  'log_filtered_scaled', 'outliers_removed', 'top_genes', "after pca"]
            )
        sample_qc_report = create_sample_qc_table(
            sample_ID = pd.Series(np.arange(cov_table.shape[0])),
            x_samples_corr = x_samples_corr, 
            cov_table = cov_table
        )
     
        # . colect reports
        global_qc[f"test{i}"]=global_qc_report
        sample_qc[f"test{i}"]=sample_qc_report
        gene_qc[f"test{i}"]=None # not done, because it was a trouble, with making alternative function - out of the scope for that task        
        
        # . verbose
        verbose_info(f"test{i}",global_qc_report, x_samples_accepted, x_samples_removed, ttest_results, x_transf_deg, verbose=verbose)

    # COLLECT ALL RESULTS
    data = {
        "tpm_data" : tpm_data,
        "target_data" : target_data, 
        "covariants_data" : covariants_data
        }
    qc = {
        "global_qc" : global_qc,
        "sample_qc" : sample_qc, 
        "gene_qc" : gene_qc
        }
    
    return data, qc
        