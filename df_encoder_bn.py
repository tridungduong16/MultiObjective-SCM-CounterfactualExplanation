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
import argparse



if __name__ == "__main__":

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev) 

    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='siag')
    parser.add_argument('--version', type=str, default='full')
    parser.add_argument('--emb_size', type=int, default=512)

    args = parser.parse_args()
    name = args.name
    version = args.version
    emb_size = args.emb_size
    
    # name = 'siag'
    # name = 'simple_bn'
    # version = 'full'
    # version = 'positive'
    # version = 'negative'
    
    seed = 0
    torch.manual_seed(seed)   
 
    """Load dataset """
    DATA_PATH = cf.BASE_PATH.format('cf-feasibility/generativecf/data/data-generativecf/{}')
    
    if name == 'simple_bn':
        SIMPLE_BN = DATA_PATH.format('bn1.csv')
        data_df = pd.read_csv(SIMPLE_BN)
        data_df = data_df[['x1', 'x2', 'x3', 'y']]
        col = ['x1', 'x2', 'x3']
        for c in col:
            data_df[c] = (data_df[c]-data_df[c].min())/(data_df[c].max()-data_df[c].min())
        if version == 'positive':
            data_df = data_df[data_df.y == 1].reset_index()
        elif version == 'negative':
            data_df = data_df[data_df.y == 0].reset_index()
        features = data_df[col]
    elif name == 'siag':
        SIMPLE_BN = DATA_PATH.format('sangiovese.csv')
        data_df = pd.read_csv(SIMPLE_BN)
        col = ['SproutN', 'BunchN', 'WoodW', 'SPAD06',
       'NDVI06', 'SPAD08', 'NDVI08', 'Acid', 'Potass', 'Brix', 'pH', 'Anthoc',
       'Polyph']
        for c in col:
            data_df[c] = (data_df[c]-data_df[c].min())/(data_df[c].max()-data_df[c].min())   
        data_df.loc[data_df['GrapeW'] > 0, 'outcome'] = 1
        data_df.loc[data_df['GrapeW'] <= 0, 'outcome'] = 0
        if version == 'positive':
            data_df = data_df[data_df.outcome == 1].reset_index()
        elif version == 'negative':
            data_df = data_df[data_df.outcome == 0].reset_index()

        features = data_df[col]
    
    """Select feature for training"""
    # features = data_df.iloc[:,:-1]
    # train = data_df.sample(frac=.8, random_state=42)
    # test = data_df.loc[~data_df.index.isin(train.index)]

    X_train = features
    X_val = features

    
    """Model architecture"""
    ae_model = AutoEncoder(
        encoder_layers = [512, 512, 512], #model architecture
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

    torch.save(ae_model, cf.FINAL_MODEL_PATH.format('/' + name + '/dfencoder_{}_{}.pt'.format(version,emb_size)))


    