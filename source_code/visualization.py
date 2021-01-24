#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:26:44 2021

@author: trduong
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import source_code.configuration_path as cf
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)
    
def plot_bartchart(axis, dataframe, x_value, y_value, hue_value, name, rc):
    
    sns.set_context("notebook", rc=rc)
    ax_plt = sns.barplot(x="Method", y="Value", hue="Dataset", palette="rocket_r", data=dataframe, ax=axis)


    axis.set_ylabel("Value",fontsize= 14.5) 
    axis.set_xticklabels(ax_plt.get_xmajorticklabels(),fontsize= 14.5) 
    axis.set_yticklabels(ax_plt.get_ymajorticklabels(),fontsize= 14.5) 

    # plt.setp(ax_plt.get_yticklabels())
    axis.set_title(name,fontsize= font_z) # title of plot
    
    ax_plt.title.set_text(name)
    ax_plt.set(xlabel=None)
    
    change_width(ax_plt, .24)
    
    
    
    plt.setp(ax_plt.get_legend().get_texts(), fontsize='12') # for legend text
    plt.setp(ax_plt.get_legend().get_title(), fontsize='12') # for legend title
    box = ax_plt.get_position()
    ax_plt.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position    


if __name__ == '__main__':
    dataset_name = 'siag'
    df_siag = pd.read_csv(cf.EVALUATION_PATH.format(dataset_name + '.csv'))
    dataset_name = 'simple_bn'
    df_simple_bn = pd.read_csv(cf.EVALUATION_PATH.format(dataset_name + '.csv'))
    dataset_name = 'adult'
    df_adult = pd.read_csv(cf.EVALUATION_PATH.format(dataset_name + '.csv'))
    
    df_siag = df_siag.rename(columns={"metric": "Metric", "method": "Method", "value" : "Value"})
    df_simple_bn = df_simple_bn.rename(columns={"metric": "Metric", "method": "Method", "value" : "Value"})
    df_adult = df_adult.rename(columns={"metric": "Metric", "method": "Method", "value" : "Value"})
    
    
    df_siag['Dataset'] = 'Sangiovese'
    df_simple_bn['Dataset'] = 'Simple-BN'
    df_adult['Dataset'] = 'Adult'
    
    frames = [df_siag, df_simple_bn, df_adult]
    df_whole = pd.concat(frames)
    
    method_map  = {
        'certifai': 'CERTIFAI',
        'mad' : 'MAD',
        'mobj' : 'MulObj_ProSCM'
    }
    
    df_whole['Method'] = df_whole['Method'].map(method_map)
    
    df_whole['Value'] = np.where(df_whole['Metric'] == 'target valid',df_whole['Value'] * 100,df_whole['Value'])
    df_whole['Value'] = np.where(df_whole['Metric'] == 'causal validity',df_whole['Value'] * 100,df_whole['Value'])
    
    valid_cf = df_whole[df_whole['Metric'] == 'target valid']
    causal_valid = df_whole[df_whole['Metric'] == 'causal validity']
    con_proximity = df_whole[df_whole['Metric'] == 'continuous proximity']
    cat_proximity = df_whole[df_whole['Metric'] == 'categorical proximity']
    IM1 = df_whole[df_whole['Metric'] == 'IM1']
    IM2 = df_whole[df_whole['Metric'] == 'IM2']

    """Set up hyperparameters"""
    font_z  = 20
    rc={'font.size': font_z, 'axes.labelsize': font_z, 'legend.fontsize': font_z, 
        'axes.titlesize': font_z, 'xtick.labelsize': font_z, 'ytick.labelsize': font_z}
    
    """Visualize proximity"""
    with plt.style.context('seaborn-darkgrid'):
        """Get the subplot"""
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
                        
        x_value = 'Method'
        y_value = 'Value'
        hue_value = 'Dataset'
        
        name = 'Categorical proximity'
        plot_bartchart(axes[0], cat_proximity, x_value, y_value, hue_value, name, rc)
        name = 'Continuous proximity'
        plot_bartchart(axes[1], con_proximity, x_value, y_value, hue_value, name, rc)
        
        


    fig.savefig(cf.FIGURE_PATH.format('proximity.png'), bbox_inches = 'tight')

    """Visualize interpertability score"""    
    with plt.style.context('seaborn-darkgrid'):
        """Get the subplot"""
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        sns.set(rc=rc)
        sns.set(font_scale = 2)
                        
        x_value = 'Method'
        y_value = 'Value'
        hue_value = 'Dataset'

        name = 'IM1'
        plot_bartchart(axes[0], IM1, x_value, y_value, hue_value, name, rc)
        name = 'IM2'
        plot_bartchart(axes[1], IM2, x_value, y_value, hue_value, name, rc)
        
    fig.savefig(cf.FIGURE_PATH.format('IM.png'), bbox_inches = 'tight')
    
    """Visualize validity score"""
    with plt.style.context('seaborn-darkgrid'):
        """Get the subplot"""
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
                        
        x_value = 'Method'
        y_value = 'Value'
        hue_value = 'Dataset'
        
        name = 'Target-Class Validity'
        sns.set_context("notebook", rc=rc)
        ax_plt = sns.barplot(x="Method", y="Value", hue="Dataset", palette="rocket_r", data=valid_cf, 
                              ax=axes)
        ax_plt.title.set_text(name)
        ax_plt.set(xlabel=None)
        ax_plt.yaxis.set_major_formatter(mtick.PercentFormatter())
        axes.set_title(name,fontsize= font_z) # title of plot
        axes.set_xticklabels(ax_plt.get_xmajorticklabels(),fontsize= 14.5) 
        axes.set_yticklabels(ax_plt.get_ymajorticklabels(),fontsize= 14.5) 
        axes.set_ylabel("Value",fontsize= 14.5) 
        
        change_width(ax_plt, .24)
        box = ax_plt.get_position()
        ax_plt.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position    
        ax_plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        plt.setp(ax_plt.get_legend().get_texts(), fontsize='12') # for legend text
        plt.setp(ax_plt.get_legend().get_title(), fontsize='12') # for legend title    

    fig.savefig(cf.FIGURE_PATH.format('target_valid.png'), bbox_inches = 'tight')

    """Visualize validity score"""
    with plt.style.context('seaborn-darkgrid'):
        """Get the subplot"""
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
                        
        x_value = 'Method'
        y_value = 'Value'
        hue_value = 'Dataset'
        
        name = 'Causal-Constraint Validity'
        sns.set_context("notebook", rc=rc)
        ax_plt = sns.barplot(x="Method", y="Value", hue="Dataset", palette="rocket_r", data=causal_valid, 
                              ax=axes)
        
        axes.set_title(name,fontsize= font_z) # title of plot
        axes.set_xticklabels(ax_plt.get_xmajorticklabels(),fontsize= 14.5) 
        axes.set_yticklabels(ax_plt.get_ymajorticklabels(),fontsize= 14.5) 
        axes.set_ylabel("Value",fontsize= 14.5) 

        ax_plt.set(xlabel=None)
        ax_plt.yaxis.set_major_formatter(mtick.PercentFormatter())

        change_width(ax_plt, .24)
        box = ax_plt.get_position()
        ax_plt.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position    
        ax_plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
        plt.setp(ax_plt.get_legend().get_texts(), fontsize='12') # for legend text
        plt.setp(ax_plt.get_legend().get_title(), fontsize='12') # for legend title    
        
    fig.savefig(cf.FIGURE_PATH.format('causal_valid.png'), bbox_inches = 'tight')
    
    
    
    # # Initialize a grid of plots with an Axes for each walk
    # grid = sns.FacetGrid(causal_valid, col="walk", hue="walk", col_wrap=2, size=5,
    #         aspect=1)
    
    # # Draw a bar plot to show the trajectory of each random walk
    # bp = grid.map(sns.barplot, "step", "position", palette="Set3")
    
    # # The color cycles are going to all the same, doesn't matter which axes we use
    # Ax = bp.axes[0]
    
    # # Some how for a plot of 5 bars, there are 6 patches, what is the 6th one?
    # Boxes = [item for item in Ax.get_children()
    #          if isinstance(item, matplotlib.patches.Rectangle)][:-1]
    
    # # There is no labels, need to define the labels
    # legend_labels  = ['a', 'b', 'c', 'd', 'e']
    
    # # Create the legend patches
    # legend_patches = [matplotlib.patches.Patch(color=C, label=L) for
    #                   C, L in zip([item.get_facecolor() for item in Boxes],
    #                               legend_labels)]
    
    # # Plot the legend
    # plt.legend(handles=legend_patches)

 