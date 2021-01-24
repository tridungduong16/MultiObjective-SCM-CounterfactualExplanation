#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 12:55:09 2021

@author: trduong
"""
import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import pygad
import numpy as np
import pandas as pd
import copy
import logging 
import argparse
import pickle
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing


import torch 
import torch.nn as nn

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader
from dfencoder.autoencoder import AutoEncoder


import autograd.numpy as anp

from pymoo.model.problem import Problem
from pymoo.util.normalization import normalize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
import matplotlib.pyplot as plt
from pymoo.util import plotting

# from source_code.multi_objective_optimization import continous_dist
# from source_code.multi_objective_optimization import two_cate_dist, cat_representation_dist
# from source_code.multi_objective_optimization import proto_loss, constraints_loss
# from source_code.multi_objective_optimization import compute_yloss
import source_code.configuration_path as cf
from source_code.prototype import find_proto, get_pos_neg_latent
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
# def causal_loss(linear_model, xcf, x0):
#     x3 = linear_model.predict(xcf[:2].reshape(1,-1))
#     loss = ((x3-x0[2]))
#     return loss

"""Compute the prototype loss function"""
def proto_loss(zcf, proto):   
    criterion = nn.MSELoss()
    proto_loss = criterion(proto.reshape(-1), zcf)
    proto_loss = proto_loss.detach().cpu().numpy()
    return proto_loss**2

"""
Number of variable: 30
Number of objective function: 2
"""


def pure_distance(x0, xcf):
    return mean_squared_error(xcf, x0)

def cross_entropy(pred_model, xcf, targets=1,epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions. 
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray        
    Returns: scalar
    """
    
    predictions = pred_model.predict(np.array(xcf).reshape(1,-1))
    predictions = predictions.reshape(-1,1)
    targets = np.array(targets).reshape(-1,1)
    
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = np.sum(targets*np.log(predictions+1e-9))/N
    return ce


class CF4(Problem):
    def __init__(self, x0, 
                 pred_model,
                 dfencoder_model,
                 scm_model,
                 proto,
                 col, 
                 n_var=3, 
                 **kwargs):
        super().__init__(n_var=n_var, 
                         n_obj=3, 
                         n_constr=0, 
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         xl=np.array([0,0,0]), 
                         xu=np.array([1,1,1]),
                         
                         **kwargs)
        

        self.x0 = x0
        self.pred_model = pred_model
        self.dfencoder_model = dfencoder_model
        self.scm_model = scm_model
        self.col = col
        self.proto = proto
        
    def _evaluate(self, xcf, out, *args, **kwargs):
        xcf[2] = self.scm_model.predict(xcf[:2].reshape(1,-1))

        # dist = np.mean(np.abs(xcf - x0))        
        dist = pure_distance(self.x0,xcf)
        yloss = cross_entropy(self.pred_model, xcf)
        # closs = causal_loss(self.scm_model, xcf, self.x0)
        
        df_store = pd.DataFrame(columns = self.col)
        df_store.loc[0] = xcf
        zcf = dfencoder_model.get_representation(df_store)[0]
        
        ploss = proto_loss(zcf, self.proto)
        
        out["F"] = anp.column_stack([yloss, dist, ploss])
        # out["F"] = anp.column_stack([yloss, dist, closs, ploss])
        

if __name__ == "__main__":

    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev)  

    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='different_instance')
    parser.add_argument('--n_instance', type=int, default=10)
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    
        
        
    args = parser.parse_args()
    mode = args.mode
    n_instance = args.n_instance
    n_sample = args.n_sample
    emb_size = args.emb_size
    seed = args.seed
    
    DATA_PATH = cf.BASE_PATH.format('cf-feasibility/generativecf/data/data-generativecf/{}')
    SIMPLE_BN = DATA_PATH.format('bn1.csv')

    """Load data"""
    data = pd.read_csv(SIMPLE_BN)
    
    col = ['x1', 'x2', 'x3']
    
    data = data[['x1', 'x2', 'x3', 'y']]
    data['x1'] = (data['x1']-data['x1'].min())/(data['x1'].max()-data['x1'].min())
    data['x2'] = (data['x2']-data['x2'].min())/(data['x2'].max()-data['x2'].min())
    data['x3'] = (data['x3']-data['x3'].min())/(data['x3'].max()-data['x3'].min())

    features = data[['x1', 'x2', 'x3']]
    
    df_original = pd.DataFrame(columns = list(data.columns))
    df_cf = pd.DataFrame(columns = list(data.columns))
    
    
    """Build model """
    # linear_model = LinearRegression().fit(data[['x1', 'x2']], data['x3'])
    linear_model = ElasticNet(random_state=0, l1_ratio=0.75).fit(data[['x1', 'x2']], data['x3'])
    
    # linear_model = GradientBoostingRegressor(random_state=0).fit(data[['x1', 'x2']], data['x3'])
    
    # pred_model = LogisticRegression(random_state=0).fit(data[['x1', 'x2', 'x3']], data['y'])
    
    # with open(cf.MODEL_FINAL_PATH.format('bn_prediction.pkl'), 'wb') as f:
    #     pickle.dump(pred_model, f)
    with open(cf.MODEL_FINAL_PATH.format('bn_linear.pkl'), 'wb') as f:
        pickle.dump(linear_model, f)

    """Load model"""
    version = 'full'
    name = 'simple_bn'
    pred_model = pickle.load(open(cf.MODEL_FINAL_PATH.format('bn_prediction.pkl'), 'rb'))
    linear_model = pickle.load(open(cf.MODEL_FINAL_PATH.format('bn_linear.pkl'), 'rb'))
    print("Load autoencoder model from {}".format(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size))))

    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size)))
                           

    """Get prediction and negative index"""
    y_prediction = pred_model.predict(features.values)
    neg_index = np.where(y_prediction == 0)[0] 

    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)    
    y_prediction = torch.Tensor(y_prediction)
    pos_z, neg_z = get_pos_neg_latent(y_prediction, z_representation)
    
    """Setup for genetic algorithm"""
    mask = ["real","real","real"]

    
    ORG_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/final_result/result/original/{}"
    data.iloc[neg_index,:].to_csv(ORG_PATH.format('full_negative_simple_bn.csv'), index = False)
    
    # sys.exit(1)

    df_store = pd.DataFrame(columns = list())
    loc_ = 0
    for i in range(n_sample):
        print("Sample index {}".format(i))
        index_ = neg_index[i]
        x0 = features.loc[index_].values
        z0 = z_representation[index_]        

        """Find prototype"""
        k_instance= n_instance
        pos_proto, neg_proto = find_proto(z0.reshape(1,-1), pos_z, neg_z, k_instance)
        
        print("Define problem")
        problem = CF4(x0, 
              pred_model, 
              dfencoder_model,
              linear_model,
              pos_proto,
              col
              )
        
        print("Define algorithm")
        algorithm = NSGA2(
                pop_size=200,
                n_offsprings=50,
                sampling=get_sampling("real_random"),
                crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
                # crossover=get_crossover("real_de"),
                mutation=get_mutation("real_pm", eta=3.0),
                eliminate_duplicates=True)
        
        print("Run optimization progress")
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 50),
                        save_history=True,
                        seed = seed,
                        verbose=True)
        

        df_store = pd.DataFrame(columns = list(features.columns))
        
        x_valid = 0
        current_dist = 9999
        
        for j in range(len(res.X)):
            xcf = res.X[j]
            ycf = pred_model.predict(np.array(xcf).reshape(1,-1))[0]
            
            if ycf == 1:
                # if all(x >=0  for x in (xcf-x0)) or all(x < 0  for x in (xcf-x0)):
                if pure_distance(x0, xcf) <= current_dist:
                    current_dist = pure_distance(x0, xcf)
                    x_valid = xcf
                    index_ = j
        if type(x_valid) != int:
            print("Found valid solution")
            df_cf.loc[i, :-1] = x_valid
            df_cf.loc[i, 'y'] = pred_model.predict(np.array(x_valid).reshape(1,-1))[0]
        else: 
            df_cf.loc[i, :-1] = xcf
            df_cf.loc[i, 'y'] = pred_model.predict(np.array(xcf).reshape(1,-1))[0]

        
        df_original.loc[i, :-1] =  x0
        df_original.loc[i, 'y'] = pred_model.predict(np.array(x0).reshape(1,-1))[0]
        
        # /data/trduong/counterfactual-explanation-research.git/my-algorithm/final_result/result/simple_bn
        # print(df_cf)
        print(df_cf)
        print(df_original)
        print("\n")
        print(df_cf - df_original)
    print("Output file to ", cf.FINAL_RESULT_PATH.format("simple_bn/"+ mode + "/mobj-ninstance-{}-nsample-{}-size-{}.csv".format(n_instance, n_sample,emb_size)))

    df_original.to_csv(cf.FINAL_RESULT_PATH.format("simple_bn/original.csv"),
                              index = False)
    df_cf.to_csv(cf.FINAL_RESULT_PATH.format("simple_bn/"+ mode + "/mobj-ninstance-{}-nsample-{}-size-{}-seed-{}.csv".format(n_instance, n_sample, emb_size, seed)),
                             index = False)

