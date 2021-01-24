#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:59:22 2020

@author: trduong
"""
import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import source_code.configuration_path as cf
from source_code.helpers_algo import load_encoder_data
from source_code.evaluation import con_proximity, cat_proximity

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

import torch 


def convert_to_label(df, d):
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        df.loc[:,c] = list(le.transform(df.loc[:,c].values))
    return df 

def im1(xcf, ae1, ae0):
    return np.linalg.norm(xcf-ae1) / (np.linalg.norm(xcf-ae0) + 0.05)

def im2(ae1, ae_full, xcf):
    return np.linalg.norm(ae1-ae_full) / (np.linalg.norm(xcf, ord=1) + 0.05)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)
                
if __name__ == "__main__":
    """Load dataset """
    d, encoded_data = load_encoder_data()
    data_df = d.data_df.drop(columns = ['income'])
    
    """Load data and two baselines"""
    version = 0
    df_dice_original = pd.read_csv(cf.FINAL_RESULT_PATH.format('dice_original.csv'))
    df_dice_counterfactul = pd.read_csv(cf.FINAL_RESULT_PATH.format('dice_counterfactual.csv'))
    
    df_genetic_original = pd.read_csv(cf.FINAL_RESULT_PATH.format('dice_original.csv'.format(version)))
    df_genetic_counterfactual = pd.read_csv(cf.FINAL_RESULT_PATH.format('multi_objective_cf.csv'.format(version)))
    
    df_mad_original = pd.read_csv(cf.FINAL_RESULT_PATH.format('dice_original.csv'))
    df_mad_counterfactuall = pd.read_csv(cf.FINAL_RESULT_PATH.format('genetic_counterfactual_mad.csv'))

    """Load auto encoder model"""
    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format('dfencoder_name.pt'))
    dfencoder_model_pos = torch.load(cf.FINAL_MODEL_PATH.format('dfencoder_name_positive.pt'))
    dfencoder_model_neg = torch.load(cf.FINAL_MODEL_PATH.format('dfencoder_name_negative.pt'))
    
    """Get representation"""
    z_dice_cf = dfencoder_model.get_representation(df_dice_counterfactul)
    z_dice_cf_pos = dfencoder_model_pos.get_representation(df_dice_counterfactul)
    z_dice_cf_neg = dfencoder_model_neg.get_representation(df_dice_counterfactul)

    z_mad_cf = dfencoder_model.get_representation(df_mad_original)
    z_mad_cf_pos = dfencoder_model_pos.get_representation(df_mad_original)
    z_mad_cf_neg = dfencoder_model_neg.get_representation(df_mad_original)
    
    # sys.exit("Age less than 18")     
    
    """Calculate proximity"""
    df_evaluation = pd.DataFrame()
    df_evaluation['method'] = ''
    df_evaluation['Metric'] = ''
    df_evaluation['value'] = 0
    
    con_indexes = [0,7]
    cat_indexes = [1,2,3,4,5,6]
    n_sample = 10
    loc_ = 0
    
    col = ['race', 
            'gender',
            'age',
            'workclass',
            'education',
            'marital_status',
            'occupation']
    regularization = 0.05
    """Evaluation in dice"""
    
    term_original = df_dice_original[col]
    term_cf = df_dice_counterfactul[col]    
    term_original = convert_to_label(term_original, d)
    term_cf = convert_to_label(term_cf, d)
    
    for i in range(n_sample):
        x0 = df_dice_original.loc[i].values
        xcf = df_dice_counterfactul.loc[i].values
        
        """Compute category proximity"""
        cat_prox = cat_proximity(x0, xcf, cat_indexes)
        df_evaluation.loc[loc_,'method'] = 'dice'
        df_evaluation.loc[loc_,'Metric'] = 'cat_proximity'
        df_evaluation.loc[loc_,'value'] = cat_prox
        loc_ += 1
        
        """Compute continuous proximity"""
        con_prox = con_proximity(x0, xcf, con_indexes)
        df_evaluation.loc[loc_,'method'] = 'dice'
        df_evaluation.loc[loc_,'Metric'] = 'continuos_proximity'
        df_evaluation.loc[loc_,'value'] = con_prox
        loc_ += 1
        
        """Compute IM1 and IM2"""
        rec_cf = dfencoder_model.decode_to_df(z_dice_cf[i].reshape(1,-1))
        rec_pos_cf = dfencoder_model_pos.decode_to_df(z_dice_cf_pos[i].reshape(1,-1))
        rec_neg_cf = dfencoder_model_neg.decode_to_df(z_dice_cf_neg[i].reshape(1,-1))
        
        rec_cf = convert_to_label(rec_cf, d)
        rec_pos_cf = convert_to_label(rec_pos_cf, d)
        rec_neg_cf = convert_to_label(rec_neg_cf, d)
        
        im1_val = im1(term_cf.loc[i].values, rec_pos_cf.loc[0].values, rec_neg_cf.loc[0].values)
        im2_val= im2(rec_pos_cf.loc[0].values, rec_cf.loc[0].values, term_cf.loc[i].values)
        
        # df_evaluation.loc[loc_,'method'] = 'dice'
        # df_evaluation.loc[loc_,'Metric'] = 'IM1'
        # df_evaluation.loc[loc_,'value'] = im1_val
        # loc_ += 1        

        # df_evaluation.loc[loc_,'method'] = 'dice'
        # df_evaluation.loc[loc_,'Metric'] = 'IM2'
        # df_evaluation.loc[loc_,'value'] = im2_val
        # loc_ += 1  
        
        """Compute valid"""
        df_evaluation.loc[loc_,'method'] = 'dice'
        df_evaluation.loc[loc_,'Metric'] = 'valid_cf'
        df_evaluation.loc[loc_,'value'] = 1 if xcf[-1] >= 0.5 else 0
        loc_ += 1          
        
        
    """Evaluation in mad"""
    for i in range(n_sample):
        x0 = df_mad_original.loc[i].values
        xcf = df_mad_counterfactuall.loc[i].values
        
        """Compute categorical proximity"""
        cat_prox = cat_proximity(x0, xcf, cat_indexes)
        df_evaluation.loc[loc_,'method'] = 'mad'
        df_evaluation.loc[loc_,'Metric'] = 'cat_proximity'
        df_evaluation.loc[loc_,'value'] = cat_prox
        loc_ += 1
        
        """Compute continuous proximity"""
        con_prox = con_proximity(x0, xcf, con_indexes)
        df_evaluation.loc[loc_,'method'] = 'mad'
        df_evaluation.loc[loc_,'Metric'] = 'continuos_proximity'
        df_evaluation.loc[loc_,'value'] = con_prox
        loc_ += 1
        
        
        """Compute valid"""
        df_evaluation.loc[loc_,'method'] = 'mad'
        df_evaluation.loc[loc_,'Metric'] = 'valid_cf'
        df_evaluation.loc[loc_,'value'] = 1 if xcf[-1] >= 0.5 else 0
        loc_ += 1     
        
        
    """Evaluation in genetic_proto"""
    for i in range(n_sample):
        x0 = df_genetic_original.loc[i].values
        xcf = df_genetic_counterfactual.loc[i].values
        
        """Compute categorical proximity"""
        cat_prox = cat_proximity(x0, xcf, cat_indexes)
        df_evaluation.loc[loc_,'method'] = 'genetic_proto_' + str(version)
        df_evaluation.loc[loc_,'Metric'] = 'cat_proximity'
        df_evaluation.loc[loc_,'value'] = cat_prox
        loc_ += 1
        
        """Compute continuous proximity"""
        con_prox = con_proximity(x0, xcf, con_indexes)
        df_evaluation.loc[loc_,'method'] = 'genetic_proto_' + str(version)
        df_evaluation.loc[loc_,'Metric'] = 'continuos_proximity'
        df_evaluation.loc[loc_,'value'] = con_prox
        loc_ += 1
    
        """Compute valid"""
        df_evaluation.loc[loc_,'method'] = 'genetic_proto_' + str(version)
        df_evaluation.loc[loc_,'Metric'] = 'valid_cf'
        df_evaluation.loc[loc_,'value'] = 1 if xcf[-1] >= 0.5 else 0
        loc_ += 1    
        
    final_result = df_evaluation.groupby(['method', 'Metric']).mean().reset_index()
    valid_cf_result = final_result[final_result['Metric'] == 'valid_cf']
    proximity_result = final_result[final_result['Metric'] != 'valid_cf']
    
    with plt.style.context('seaborn-darkgrid'):
        """Get the subplot"""
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        
        """Set up hyperparameters"""
        rc={'font.size': 32, 'axes.labelsize': 40, 'legend.fontsize': 16, 
        'axes.titlesize': 32, 'xtick.labelsize': 40, 'ytick.labelsize': 40}
        
        sns.set(rc=rc)
        sns.set(font_scale = 2)
                
        columncounts = final_result.value.values
        
        """Plot the categorical proximity and continuous proximity"""        
        ax1 = sns.barplot(x='method', 
                        y='value', 
                        hue='Metric', 
                        palette="viridis",
                        data=proximity_result,
                        ax=axes[0])
        ax1.set_ylabel("Value",fontsize=18)    
        ax1.set_xticklabels(ax1.get_xmajorticklabels(), fontsize = 18)  
        plt.setp(ax1.get_yticklabels(), fontsize=18)
        ax1.set(xlabel=None)
        
        change_width(ax1, .24)
        plt.setp(ax1.get_legend().get_texts(), fontsize='10') # for legend text
        plt.setp(ax1.get_legend().get_title(), fontsize='10') # for legend title
    
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


        """Plot the %target-class valid counterfactual"""
        ax2 = sns.barplot(x='method', 
                        y='value', 
                        hue='Metric', 
                        palette="viridis",
                        data=valid_cf_result,
                        ax=axes[1])
        ax2.set_ylabel("Value",fontsize=18)    
        ax2.set_xticklabels(ax2.get_xmajorticklabels(), fontsize = 18)  
        plt.setp(ax2.get_yticklabels(), fontsize=18)
        plt.legend(loc='upper left')

        ax2.set(xlabel=None)
        
        change_width(ax2, .24)
        plt.setp(ax2.get_legend().get_texts(), fontsize='10') # for legend text
        plt.setp(ax2.get_legend().get_title(), fontsize='10') # for legend title

        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position        
        
        
        """Save the plot"""
        fig.savefig(cf.FIGURE_PATH.format('performance.png'), bbox_inches = 'tight')

 

    