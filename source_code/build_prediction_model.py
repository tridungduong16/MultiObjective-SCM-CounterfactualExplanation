#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:50:02 2020

@author: trduong
"""

## import sys path
import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

## import external librar
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm
import pandas as pd
import argparse

## import internal library
import source_code.configuration_path as cf
from source_code.helpers_algo import load_encoder_data
from source_code.auto_encoder import autoencoder
# from dfencoder.autoencoder import AutoEncoder


class DatasetAdult(Dataset):
    
    def __init__(self, data, transform=None):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image = self.data[index, :-1]
        label = self.data[index, -1]
        return image, label
    
class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 128)
        self.fc2 = nn.Linear(128, 64)
        self.last_layer = nn.Linear(64, 1)
        self.dp = nn.Dropout(0.5)        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dp(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.dp(x)
        
        x = self.last_layer(x)        
        x = self.sigmoid(x)
        return x
        

def train_prediction_model(train_loader,model):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 50
    for epoch in tqdm(range(epochs)):
        total = 0
        correct = 0
        running_loss = 0
        model.eval()
        for local_batch, local_labels in (train_loader):
            local_batch = local_batch.type(torch.FloatTensor).to(device)
            local_labels = local_labels.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()
            outputs = model(local_batch)
            predicted = torch.ge(outputs, 0.5).int()
            outputs = outputs.reshape(-1)
            loss = criterion(outputs, local_labels.detach())
            running_loss += loss 
            total += local_labels.size(0)
            cor = torch.eq(predicted.reshape(-1),local_labels).int().sum()
            correct += cor
            loss.backward(retain_graph=True)
            optimizer.step()
        accuracy = correct / total
        epoch_loss = running_loss / total
        if epoch % 10 == 0:
            print("\n Epoch {}, Accuracy {:.4f}, Loss {:.4f}".format(epoch,accuracy,epoch_loss))
    return model

if __name__ == "__main__":     
    #Argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    args = parser.parse_args()

    seed = 0
    torch.manual_seed(seed)  
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev)  
    
    df = pd.read_csv(cf.DATA_PATH.format("label_encoder_data.csv"))
    dfencoder = torch.load(cf.MODEL_FINAL_PATH.format('dfencoder.pt'))
    
    # sys.exit("Error message")
    
    z_representation = dfencoder.get_representation(df.iloc[:,:-1])
    label = df['income'].values
    label = torch.Tensor(label).to(device).reshape(-1,1)
    train_data = torch.hstack((z_representation,label))
    train_dataset = DatasetAdult(train_data)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    input_shape = z_representation.shape[1]
    pred_model = Net(input_shape)
    pred_model.to(device)
    
    print("Traing prediction model...")
    pred_model = train_prediction_model(train_loader,pred_model)
    # torch.save(model.state_dict(), cf.MODEL_PATH_MAIN.format("dfencoder_pred_model.pth"))
    torch.save(pred_model, cf.MODEL_FINAL_PATH.format('dfencoder_pred_model.pt'))
