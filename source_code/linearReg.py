#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 19:58:55 2018

Linear Regression in PyTorch

@author: mohak
"""

#Step1: import the libraries and data
import sys
sys.path.insert(1, '/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm')

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from source_code.helpers_algo import load_encoder_data
from sklearn import preprocessing 


#Step2: Create a class
class LinearReg(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearReg,self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.linear(x)
        out = self.softmax(out)
        return out
    
    
d, data = load_encoder_data()



children = ['education_Assoc', 'education_Bachelors',
       'education_Doctorate', 'education_HS-grad', 'education_Masters',
       'education_Prof-school', 'education_School', 'education_Some-college']

parents = ['age', 
          'gender_Female', 'gender_Male', 
          'race_Other', 'race_White',
          'marital_status_Divorced', 'marital_status_Married','marital_status_Separated',
          'marital_status_Single','marital_status_Widowed'
          ]

xTrain = Variable(torch.Tensor(data[parents].values))
yTrain = Variable(torch.Tensor(data[children].values))


criterion = nn.CrossEntropyLoss()
    
input_dim = xTrain.shape[1]
output_dim = yTrain.shape[1]
model = LinearReg(input_dim, output_dim)
if torch.cuda.is_available():
    model.cuda()

#Step4: Instanciate loss class

#Step5: Instantiate the optimizer class
learning_rate=0.001
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Step6: Epochs for loop
epochs=40
for epoch in range(epochs):
    epoch = epoch+1
    inp = xTrain
    out = yTrain
    yHat = model(xTrain)
    
    _, predicted_labels = yHat.max(dim=1)
    _, labels = out.max(dim=1)
    er = criterion(yHat,labels)
    optim.zero_grad()
    er.backward()
    optim.step()
    
    cor = torch.eq(predicted_labels,labels).int().sum()
    acc = cor / yTrain.shape[0]
    
    if(epoch%10==0):
        print('epochs = {0}, loss={1}, accuracy={2}'.format(epoch, er, acc))
        
        
from sklearn.metrics import confusion_matrix
confusion_matrix(labels, predicted_labels)



y_test = labels
y_pred = predicted_labels

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
"""
Use your classification model to predict some labels
Then, plot confusion matrix and classification report using below code
y_test: real labels
y_pred: predicted model labels
"""
# labels = ['Positive', 'Neutral', 'Negative'] 
# labels = [1, 0, -1] 

print(classification_report(y_test, y_pred, labels)) #classification report from sklearn
cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
plt.imshow(cnf_matrix, cmap=plt.cm.Blues) #plot confusion matrix grid
threshold = cnf_matrix.max() / 2 #threshold to define text color
for i in range(cnf_matrix.shape[0]): #print text in grid
    for j in range(cnf_matrix.shape[1]): 
        plt.text(j, i, cnf_matrix[i,j], color="w" if cnf_matrix[i,j] > threshold else 'black')
tick_marks = np.arange(len(labels)) #define labeling spacing based on number of classes
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.colorbar()
plt.tight_layout()
