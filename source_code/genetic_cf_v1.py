#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:19:10 2020

@author: trduong
"""

import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

from tqdm import tqdm 
import time
import argparse
from sklearn.preprocessing import LabelEncoder

import torch
import torch.optim as optim
import torch.nn as nn

from source_code.helpers_algo import load_encoder_data
from source_code.find_counterfactual import load_ae
import source_code.configuration_path as cf

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader


seed = 0
torch.manual_seed(seed)




if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)   

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev)  
        
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 
             'continuous_features':['age','hours_per_week'], 
             'outcome_name':'income'}
    d = DataLoader(params)
    normalized_data = d.normalize_data(dataset)
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        normalized_data[c] = le.fit_transform(normalized_data[c])
    
    
    ae_pos, ae_neg, ae_full = load_ae(d)

    
    z_representation = ae_full.encoder(features)
    input_shape = z_representation.shape[1]
    pred_model = Net(input_shape)
    pred_model.load_state_dict(torch.load(cf.MODEL_PATH_MAIN.format("pred_model.pth")))
    pred_model = pred_model.to(device)
    
    
    neg_prediction = torch.le(pred_model(z_representation), 0.5)
    
  
    normalized_data.to_csv(cf.DATA_PATH.format("label_encoder_data.csv"), index = False)


    