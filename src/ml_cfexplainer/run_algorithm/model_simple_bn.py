#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:55:09 2021

@author: trduong
"""
import sys
import pandas as pd
import argparse
import pickle
import torch
import yaml

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingClassifier


if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    """Read configuration"""
    with open('/data/trduong/multiobj-scm-cf/src/config.yml') as file:
      conf = yaml.safe_load(file)

    """Load data"""
    data = pd.read_csv(conf['data_simple_bn'])
    col = ['x1', 'x2', 'x3']
    data = data[['x1', 'x2', 'x3', 'y']]
    data['x1'] = (data['x1'] - data['x1'].min()) / (data['x1'].max() - data['x1'].min())
    data['x2'] = (data['x2'] - data['x2'].min()) / (data['x2'].max() - data['x2'].min())
    data['x3'] = (data['x3'] - data['x3'].min()) / (data['x3'].max() - data['x3'].min())
    features = data[['x1', 'x2', 'x3']]
    """Save data"""
    data.to_csv(conf['processed_simple_bn'], index = False)

    df_original = pd.DataFrame(columns=list(data.columns))
    df_cf = pd.DataFrame(columns=list(data.columns))

    """Build structural causal equation """
    linear_model = ElasticNet(random_state=0, l1_ratio=0.75).fit(data[['x1', 'x2']], data['x3'])
    with open(conf['scm_model_simple_bn'], 'wb') as f:
        pickle.dump(linear_model, f)

    linear_model = GradientBoostingClassifier(random_state=0).fit(data[['x1', 'x2', 'x3']], data['y'])
    with open(conf['prediction_model_simple_bn'], 'wb') as f:
        pickle.dump(linear_model, f)
