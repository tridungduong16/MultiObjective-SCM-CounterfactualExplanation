#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:01:01 2020

@author: trduong
"""
PATH = "/data/trduong/counterfactual-explanation-research.git/DiCE"
PATH2 = '/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm'
import sys
sys.path.insert(1, PATH)
sys.path.insert(1, PATH2)

import dice_ml
from dice_ml.utils import helpers # helper functions

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import pandas as pd
import numpy as np
import logging
import pprint 

from source_code.helpers_algo import load_encoder_data
from source_code.prototype import find_proto
from source_code.auto_encoder import autoencoder
from source_code.build_prediction_model import DatasetAdult, Net
from source_code.prototype import find_proto, get_pos_neg_latent
from source_code.evaluation import IM, con_proximity, cat_proximity
import source_code.configuration_path as cf

import sys
import json
import time


if __name__ == "__main__":  
    seed = 0
    torch.manual_seed(seed)  
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev)  

    """Load data and features"""    
    d, encoded_data = load_encoder_data()
    data = encoded_data.values    
    data_df = d.data_df.drop(columns = ['income'])
    df = pd.read_csv(cf.DATA_PATH.format('label_encoder_data.csv'))
    df = df.drop(columns = ['income'])
    
    features = torch.Tensor(data[:,:-1]).to(device)
    
    d = dice_ml.Data(dataframe=d.data_df, 
                     continuous_features=['age', 'hours_per_week'], 
                     outcome_name='income')
        
    
    """Load prediction and auto encoder model"""    
    dfencoder_model = torch.load(cf.MODEL_FINAL_PATH.format('dfencoder.pt'))
    pred_model = torch.load(cf.MODEL_FINAL_PATH.format('dfencoder_pred_model.pt'))
    
    """Setup Dice"""
    print("Run Dice")
    backend = 'PYT'
    ML_modelpath = '/data/trduong/counterfactual-explanation-research.git/my-algorithm/models/main/pred_model.pth'
    m = dice_ml.Model(model_path= ML_modelpath, backend=backend)
    d = dice_ml.Data(dataframe=d.data_df, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    exp = dice_ml.Dice(d, pred_model)
    
    # sys.exit("some error message")
    """Run dice with multiple data"""
    n_sample = 10 

    for i in range(n_sample):
        print("Sample index {}".format(i))
        print("Query instance ")
        
        query_instance = data_df.loc[i].to_dict()
        prepared_query_instance = d.prepare_query_instance(query_instance=query_instance, encode=True)
        prepared_query_instance = prepared_query_instance.iloc[0].values
        
        cat_name = d.categorical_feature_names
        encoded_cat = d.get_encoded_categorical_feature_indexes()
        zipped = list(zip(cat_name,encoded_cat))
        for z in (zipped):
            name = z[0]
            start_index, end_index = z[1][0], z[1][-1]
            index_ = np.argmax(prepared_query_instance[start_index:end_index+1])
            # print(d.encoded_feature_names[start_index:end_index+1][index_].replace(name + '_',''))
    
        
        # sys.exit("some error message")
        
        loss_diff_thres = 0.5
        start = time.time()

        
        dice_exp = exp.generate_counterfactuals(query_instance, 
                                        total_CFs=1, 
                                        desired_class="opposite",
                                        loss_diff_thres=loss_diff_thres)
        end = time.time()
        
        processed_time = end - start
        
        df_result = pd.DataFrame(columns = list(data_df.columns))
        df_result.loc[0,:] = data_df.loc[i].values
        df_result.loc[1,:] = dice_exp.final_cfs_df.loc[0,:] 
        df_result.to_csv(cf.RESULT_PATH.format("dice_{}.csv".format(i)), index = False)
        
        
    