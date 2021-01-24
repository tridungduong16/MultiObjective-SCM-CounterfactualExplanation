#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:51:29 2020

@author: trduong
"""

import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader



def load_encoder_data():
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 
             'continuous_features':['age','hours_per_week'], 
             'outcome_name':'income'}
    d = DataLoader(params)
    
    #Get one hot and normalize data
    encoded_data = d.one_hot_encode_data(dataset)
    encoded_data = d.normalize_data(encoded_data)
    ##Rearrange columns
    cols = list(encoded_data.columns)
    cols = cols[:2] + cols[3:] + [cols[2]]
    encoded_data = encoded_data[cols]

    return d, encoded_data



        