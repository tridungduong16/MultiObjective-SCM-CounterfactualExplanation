#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:06:31 2020

@author: trduong
"""

import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import torch
import source_code.configuration_path as cf

from source_code.helpers_algo import load_encoder_data
from source_code.auto_encoder import autoencoder
from source_code.build_prediction_model import Net

seed = 0
torch.manual_seed(seed)

def find_proto(original_latent, pos_latent, neg_latent, k_instance):
    ori_pos_dist = torch.cdist(original_latent, pos_latent, p=1)
    ori_neg_dist = torch.cdist(original_latent, neg_latent, p=1)
    
    _ , pos_index = torch.topk(-ori_pos_dist, k=k_instance)
    _ , neg_index = torch.topk(-ori_neg_dist, k=k_instance)
    
    pos_proto = torch.mean(pos_latent[pos_index], dim = 1)
    neg_proto = torch.mean(neg_latent[neg_index], dim = 1)
    
    proto_pos_dist = torch.cdist(original_latent, pos_proto, p=1)
    proto_neg_dist = torch.cdist(original_latent, neg_proto, p=1)
    
    # print("Positive distance ", torch.mean(proto_pos_dist, dim = 1))
    # print("Negative distance ", torch.mean(proto_neg_dist, dim = 1))
    
    
    return pos_proto, neg_proto

def get_pos_neg_latent(prediction, z_representation):
    pos_locs = torch.eq(prediction, 1.0)
    neg_locs = torch.eq(prediction, 0.0)
    
    pos_z = z_representation[pos_locs]
    neg_z = z_representation[neg_locs]
    return pos_z, neg_z

if __name__ == "__main__":
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev)  
    
    d, df = load_encoder_data()
    data = df.values    
    features = torch.Tensor(data[:,:-1]).to(device)
    
    ae_model = autoencoder(d)
    ae_model.load_state_dict(torch.load(cf.MODEL_PATH + "/adult_self_implementation_ae_model.pth"))
    ae_model = ae_model.to(device)
    z_representation = ae_model.encoder(features)
    
    input_shape = z_representation.shape[1]
    pred_model = Net(input_shape)
    pred_model.load_state_dict(torch.load(cf.MODEL_PATH + "/adult_prediction_self_implementation_ae.pth"))
    pred_model = pred_model.to(device)
    
    pred = pred_model(z_representation)
    pos_z, neg_z = get_pos_neg_latent(pred_model, ae_model, z_representation)
    
    z_sample = z_representation[0:1]
    k_instance = 10
    pos_proto, neg_proto = find_proto(z_sample, pos_z, neg_z, k_instance)