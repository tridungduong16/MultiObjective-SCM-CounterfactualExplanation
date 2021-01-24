#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:30:15 2020

@author: trduong
"""
import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import torch
import torch.nn as nn
from torch.utils.data import Dataset

seed = 0
torch.manual_seed(seed)

class autoencoder(nn.Module):
    def __init__(self,d):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(29,64),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, 64),
            # nn.ReLU(True), 
            # nn.Dropout(0.5),
            # nn.Linear(64, 12), 
            # nn.ReLU(True), 
            # nn.Dropout(0.5),
            nn.Linear(64, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(12, 64),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(64, 128),
            # nn.ReLU(True), 
            # nn.Dropout(0.5),
            nn.Linear(64,29))
        self.sig = nn.Sigmoid()
        self.d = d

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        for v in self.d.get_encoded_categorical_feature_indexes():    
            start_index = v[0]
            end_index = v[-1] + 1
            x[:,start_index:end_index] = self.sig(x[:,start_index:end_index])
        return x
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(29, 64)
        # self.batch1 = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024, 512)
        # self.batch2 = nn.BatchNorm1d(512)
        # self.fc3 = nn.Linear(512, 256)
        # self.batch3 = nn.BatchNorm1d(256)
        # self.fc4 = nn.Linear(256, 128)
        # self.batch4 = nn.BatchNorm1d(128)
        # self.fc5 = nn.Linear(128, 64)
        # self.batch5 = nn.BatchNorm1d(64)
        self.last_layer = nn.Linear(64, 1)

        self.dp = nn.Dropout(0.5)        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.batch1(x)
        # x = self.dp(x)
        
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.batch2(x)
        # x = self.dp(x)
        
        # x = self.fc3(x)
        # x = self.relu(x)
        # x = self.batch3(x)
        # x = self.dp(x)
        
        # x = self.fc4(x)
        # x = self.relu(x)
        # x = self.batch4(x)
        # x = self.dp(x)

        # x = self.fc5(x)
        # x = self.batch5(x)
        # x = self.dp(x)

        x = self.last_layer(x)        
        x = self.sigmoid(x)
        return x

class DatasetAdult(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        image = self.data.iloc[index, :-1].values
        label = self.data.iloc[index, -1]
        return image, label



