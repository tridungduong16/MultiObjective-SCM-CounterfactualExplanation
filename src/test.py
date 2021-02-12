#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:57:31 2021

@author: trduong
"""

from sklearn.metrics import log_loss
from sklearn.metrics import hinge_loss
import pandas as pd
import pickle 
import yaml
import numpy as np
from scipy.spatial.distance import cdist 


print(log_loss([1, 0],[0.8, 0]))
print(log_loss([1, 0],[0.5, 0]))
print(log_loss([1, 0],[0.4, 0]))
print(log_loss([1, 0],[0.1, 0]))

# print(hinge_loss([0],[0.1]))

"""Load configuration"""
with open('/home/trduong/Data/multiobj-scm-cf/src/config.yml') as file:
    conf = yaml.safe_load(file)

df = pd.read_csv("/data/trduong/multiobj-scm-cf/reports/result/simple_bn/result_nin-10-ns-489-size-512-seed-1.csv")
# df = pd.read_csv("/data/trduong/multiobj-scm-cf/reports/result/simple_bn/original-nsample-2.csv")

features  = ['x1', 'x2', 'x3']
pred_model = pickle.load(open(conf['prediction_model_simple_bn'], 'rb'))

x0 = df.loc[0, features].values
# prediction = pred_model.predict_proba(df[features].values)[:,1]
# print(prediction)


y_prediction = pred_model.predict(df[features].values)
pos_index = np.where(y_prediction == 1)[0]

filtered_arr  = df[df.index.isin(pos_index)].loc[0:10,features].values

Y = cdist(x0.reshape(1,-1), filtered_arr, 'euclidean')[0]
min_index = np.where(Y == min(Y))[0]

x_min = filtered_arr[min_index]




