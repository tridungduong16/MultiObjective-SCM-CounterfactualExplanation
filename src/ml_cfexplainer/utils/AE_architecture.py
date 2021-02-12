#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 16:52:27 2020

@author: trduong
"""
import torch
import torch.utils.data
from torch import nn

class autoencoder(nn.Module):
    def __init__(self,
                 index_age,
                 index_hours,
                 index_workclass,
                 index_education,
                 index_marital_status,
                 index_occupation,
                 index_race,
                 index_gender
                 ):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(29,128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128,29))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        feature_age = x[:,:self.index_age]
        feature_hours = x[:,self.index_age:self.index_hours]
        feature_workclass = self.sig(x[:,self.index_hours:self.index_workclass])
        feature_education = self.sig(x[:,self.index_workclass:self.index_education])
        feature_marital = self.sig(x[:,self.index_education:self.index_marital_status])
        feature_occupation = self.sig(x[:,self.index_marital_status:self.index_occupation])
        feature_race = self.sig(x[:,self.index_occupation:self.index_race])
        feature_gender = self.sig(x[:,self.index_race:self.index_gender])
        
        final_feature = torch.cat((feature_age, 
                                  feature_hours,
                                  feature_workclass,
                                  feature_education,
                                  feature_marital,
                                  feature_occupation,
                                  feature_race,
                                  feature_gender), dim = 1)
        
        return final_feature