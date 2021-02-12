#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:19:05 2021

@author: trduong
"""

import numpy as np
import pandas as pd
import logging
import yaml 
import sys

import torch
from pymoo.model.problem import Problem

import autograd.numpy as anp
from .distance import Distance


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class CF_Credit(Problem):
    def __init__(self, x0,
                 pred_model,
                 dfencoder_model,
                 col,
                 xl,
                 xu,
                 n_obj,
                 n_var,
                 proto = None,
                 scm_model = None,
                 con_index = None,
                 cat_index = None,
                 dict_cat_index = None,
                 **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         xl=xl,
                         xu=xu,
                         **kwargs)

        self.x0 = x0
        self.pred_model = pred_model
        self.dfencoder_model = dfencoder_model
        self.scm_model = scm_model
        self.col = col
        self.proto = proto
        self.con_index = con_index
        self.dict_cat_index = dict_cat_index

    def _evaluate(self, xcf, out, *args, **kwargs):
        """Create the distance object"""
        dist = Distance(self.x0, xcf, self.pred_model, self.dfencoder_model, con_index=self.con_index, dict_cat_index = self.dict_cat_index)

        condist = dist.continous_dist()
        yloss = dist.cross_entropy()
        catdist = dist.cat_representation_dist()

        df_store = pd.DataFrame(columns=self.col)
        df_store.loc[0] = xcf

        out["F"] = anp.column_stack([yloss, condist, catdist])
        
    
if __name__ == '__main__':
    """Load configuration"""
    with open('/home/trduong/Data/multiobj-scm-cf/src/config.yml') as file:
        conf = yaml.safe_load(file)

    """Set up logging"""
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=conf['credit_log_path'])
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    """Load data"""
    data = pd.read_csv(conf['data_credit'])

    outcome = 'default payment next month'
    feature = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    category = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    categorical_index = {'SEX' : 1, 'EDUCATION': 2, 'MARRIAGE': 3}
    
    
    data = data[feature]
        