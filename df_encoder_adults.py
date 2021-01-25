#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 10:58:19 2020

@author: trduong
"""

import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import pandas as pd
import torch

from dfencoder.autoencoder import AutoEncoder
from source_code.helpers_algo import load_encoder_data
import source_code.configuration_path as cf
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse


if __name__ == "__main__":

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev) 

    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='adult')
    parser.add_argument('--version', type=str, default='full')
    parser.add_argument('--emb_size', type=int, default=512)

    args = parser.parse_args()
    name = args.name
    version = args.version
    emb_size = args.emb_size
    # name = 'adult'
    # version = 'full'
    # version = 'positive'
    # version = 'negative'
    
    
    seed = 0
    torch.manual_seed(seed)   
 
    """Load dataset """
    d, encoded_data = load_encoder_data()
    
    data_df = d.data_df

    if version == 'positive':
        data_df = data_df[data_df.income == 1].reset_index()
    elif version == 'negative':
        data_df = data_df[data_df.income == 0].reset_index()    
    data_df.drop(columns = ['income'], inplace = True)
    data_df = data_df.reset_index(drop = True)
    data_df = d.normalize_data(data_df)
    """Load prediction and auto encoder model"""    
    # dfencoder_model = torch.load(cf.MODEL_FINAL_PATH.format('dfencoder.pt'))
        
    # pred_model = torch.load('/data/trduong/DiCE/dice_ml/utils/sample_trained_models/adult.pth')
    # pred_model.to(device)
    
    
    # sys.exit("Age less than 18")     
    
    # """Get the representation for the dataset"""
    # z_representation = dfencoder_model.get_representation(features)
    # prediction_score = pred_model(encoded_features_tensor)
    # prediction = torch.ge(prediction_score, 0.5).int()
    # prediction = prediction.reshape(-1)    
    # neg_index = (prediction == 0).nonzero()
    

    # sys.exit("Age less than 18")     
    """Convert data to category"""
    for v in d.categorical_feature_names:
        data_df[v] = pd.Categorical(data_df[v].values)
    
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        data_df[c] = le.fit_transform(data_df[c].values)

    for v in d.categorical_feature_names:
        data_df[v] = pd.Categorical(data_df[v].values)
        
    """Only consider the subset features"""    
    #df = df[['age', 'race', 'gender', 'education', 'workclass', 'marital_status']]
    col = ['age','hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'race',
       'gender']
    
    
    """Select feature for training"""
    features = data_df[col]
    train = data_df.sample(frac=.8, random_state=42)
    test = data_df.loc[~data_df.index.isin(train.index)]

    X_train = features
    X_val = test

    
    """Model architecture"""
    ae_model = AutoEncoder(
        encoder_layers = [512, 512, emb_size], #model architecture
        decoder_layers = [], #decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2, #noise parameter
        lr = 0.01,
        lr_decay=.99,
        batch_size=  512, #512
        verbose=False,
        optimizer='sgd',
        scaler='gauss_rank', #gauss rank scaling forces your numeric features into standard normal distributions 
    )
    
    """Train model"""
    ae_model.to(device)
    ae_model.fit(X_train, epochs=100, val=X_val)
 
    """Save model"""
    print("Output file to {}".format(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size))))
    torch.save(ae_model, cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size)))


    