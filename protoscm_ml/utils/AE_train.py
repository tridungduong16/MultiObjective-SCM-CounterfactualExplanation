#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:26:03 2020

@author: trduong
"""
import torch
import torch.utils.data
from torch import nn

import numpy as np 
import pandas as pd
from helpers import load_adult_income_dataset
from AE_architecture import autoencoder


def compute_loss(output, target, ae_model):
    con_loss = torch.nn.MSELoss()
    cat_loss = torch.nn.BCELoss()
    
    index_age = ae_model.index_age
    index_hours = ae_model.index_hours
    index_workclass = ae_model.index_workclass
    index_education = ae_model.index_education
    index_marital_status = ae_model.index_marital_status
    index_occupation = ae_model.index_occupation
    index_race = ae_model.index_race
    index_gender = ae_model.index_gender
    
    loss_age = con_loss(output[:,:index_age], target[:,:index_age])
    loss_hours = con_loss(output[:,index_age:index_hours], target[:,index_age:index_hours]) 
    loss_workclass = cat_loss(output[:,index_hours:index_workclass], target[:,index_hours:index_workclass])
    loss_education = cat_loss(output[:,index_workclass:index_education], target[:,index_workclass:index_education])
    loss_marital = cat_loss(output[:,index_education:index_marital_status], target[:,index_education:index_marital_status])
    loss_occupation = cat_loss(output[:,index_marital_status:index_occupation], target[:,index_marital_status:index_occupation])
    loss_race = cat_loss(output[:,index_occupation:index_race], target[:,index_occupation:index_race])
    loss_gender = cat_loss(output[:,index_race:index_gender], target[:,index_race:index_gender])

    loss = torch.mean(loss_age + 
                      loss_hours + 
                      loss_workclass + 
                      loss_education + 
                      loss_marital + 
                      loss_occupation + 
                      loss_race + 
                      loss_gender
                     )    
    
    return loss

 
dataset = load_adult_income_dataset()


n_workclass = (pd.get_dummies(dataset['workclass']).shape[1])
n_education = (pd.get_dummies(dataset['education']).shape[1])
n_marital_status = (pd.get_dummies(dataset['marital_status']).shape[1])
n_occupation = (pd.get_dummies(dataset['occupation']).shape[1])
n_race = (pd.get_dummies(dataset['race']).shape[1])
n_gender = (pd.get_dummies(dataset['gender']).shape[1])

index_age = 1
index_hours = 2
index_workclass = n_workclass + index_hours
index_education = n_education + index_workclass
index_marital_status = n_marital_status + index_education
index_occupation = n_occupation + index_marital_status
index_race = n_race + index_occupation
index_gender= n_gender + index_race

categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'gender']
preprocessed_data = pd.get_dummies(dataset, columns=categorical_features)

df_features = preprocessed_data.drop(columns = ['income'])

df_features['age'] = (df_features['age']-df_features['age'].mean()) / df_features['age'].std()
df_features['hours_per_week'] = (df_features['hours_per_week']-df_features['hours_per_week'].mean()) / df_features['hours_per_week'].std()

features = df_features.values


model = autoencoder()
criterion = nn.MSELoss()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
num_epochs = 100
batch_size = 1024

train_dataset = np.array_split(features, features.shape[0]//batch_size ,axis=0)

for epoch in range(num_epochs):
    for i in (range(len(train_dataset))):
        train_x = train_dataset[i]
        train_x = torch.tensor(train_x).float()
        train_x = torch.Tensor(train_x)
        out = model(train_x)
        loss = compute_loss(out,train_x)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # ===================log========================
    print('Epoch {}, loss:{:.4f}'.format(epoch, loss))
    
    

