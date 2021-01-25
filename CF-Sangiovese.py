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
import json


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
from sklearn.linear_model import LinearRegression

def causal_loss(linear_model, xcf, x0):
    x3 = linear_model.predict(xcf[:2].reshape(1,-1))
    loss = np.sqrt((x3-x0[2])**2)
    return loss

def pure_distance(x0, xcf):
    return mean_squared_error(x0,xcf)

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
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def list_index(elements, col):
    column = np.array(col)
    index_ = []
        
    for key in elements:
        key_index = np.where(column == key)[0]
        if len(key_index) != 0:
            index_.append(key_index[0])
    return index_

def structural_causal_equation(key, scm_model, xcf, x0, col):
    column = np.array(col)
    # print("Col ", col)
    # print("KEy ", key)
    # print(np.where(col == key))
    index_ = np.where(column == key)[0][0]
    # print()
    weight = scm_model[key]['weight'][:-1]
    values = xcf[list_index(scm_model[key]['parent'], col)]
    intercept = scm_model[key]['weight'][-1]
    
    xcf_key = np.sum(values*weight) + intercept
    x0_key = x0[index_]
    
    # print(values, '\n', weight, '\n', intercept)
    # print(xcf_key)
    
    # print(x0_key)
    
    return xcf_key - x0_key

def scm_loss(reg, x0, xcf, key, scm_model, col):
    parents = scm_model[key]['parent']
    column = np.array(col)
    index_list = list_index(parents, col)
    index_ = np.where(column == key)[0]
    # print(index_)
    # print(col)
    # print(parents)
    # print(index_list, xcf[index_list])
    xcf_key = reg.predict(xcf[index_list].reshape(1,-1))
    x0_key = x0[index_]
    # print(xcf_key - x0_key)
    return (xcf_key - x0_key)[0][0]

    #reg.predict(x0[7].reshape(-1,1))
    

"""Compute the prototype loss function"""
def proto_loss(zcf, proto):   
    criterion = nn.MSELoss()
    proto_loss = criterion(proto.reshape(-1), zcf)
    proto_loss = proto_loss.detach().cpu().numpy()
    return proto_loss**2




class CF4(Problem):
    def __init__(self, x0, 
                 pred_model,
                 dfencoder_model,
                 scm_model,
                 reg,
                 proto,
                 col, 
                 n_var=13, 
                 **kwargs):
        super().__init__(n_var=n_var, 
                         n_obj=4, 
                         n_constr=0, 
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         xl=np.array([0,0,0, 0,0,0, 0,0,0, 0,0,0, 0]), 
                         xu=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1]),
                         
                         **kwargs)
        

        self.x0 = x0
        self.pred_model = pred_model
        self.dfencoder_model = dfencoder_model
        self.scm_model = scm_model
        self.reg = reg
        self.col = col
        self.proto = proto
        
    def _evaluate(self, xcf, out, *args, **kwargs):
        dist = mean_squared_error(self.x0,xcf)
        yloss = cross_entropy(self.pred_model, xcf)
        # closs = structural_causal_equation('BunchN', scm_model, xcf, x0, col)
        key = 'BunchN'
        closs = scm_loss(self.reg, self.x0, xcf, key, self.scm_model, self.col)
        
        df_store = pd.DataFrame(columns = col)
        df_store.loc[0] = xcf
        zcf = dfencoder_model.get_representation(df_store)[0]
        
        ploss = proto_loss(zcf, self.proto)
        
        out["F"] = anp.column_stack([yloss, dist, closs, ploss])
        

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
    SIMPLE_BN = DATA_PATH.format('sangiovese.csv')

    col = ['SproutN', 'BunchN', 'WoodW', 'SPAD06',
       'NDVI06', 'SPAD08', 'NDVI08', 'Acid', 'Potass', 'Brix', 'pH', 'Anthoc',
       'Polyph']
    """Load data"""
    data = pd.read_csv(SIMPLE_BN)
    sys.exit(1)
    
    data.loc[data['GrapeW'] > 0, 'outcome'] = 1
    data.loc[data['GrapeW'] <= 0, 'outcome'] = 0
    
    for c in col:
        data[c] = (data[c]-data[c].min())/(data[c].max()-data[c].min())
        
    features = data[col]
    
    df_original = pd.DataFrame(columns = col + ['outcome'])
    df_cf = pd.DataFrame(columns = col + ['outcome'])

    """SCM model"""
    with open(DATA_PATH.format('sangiovese-scm.json')) as json_file:
        scm_model = json.load(json_file)
    
    
    """Build model """
    key = 'BunchN'
    parents = scm_model[key]['parent']
    if 'Treatment' in parents:
        parents.remove('Treatment' )


    
    reg = LinearRegression().fit(features[key].values.reshape(-1,1), features[parents].values.reshape(-1,1))
    
    # linear_model = LinearRegression().fit(data[['x1', 'x2']], data['x3'])
    # pred_model = LogisticRegression(random_state=0).fit(data[col], data['outcome'])
    
    # with open(cf.MODEL_FINAL_PATH.format('siag_prediction.pkl'), 'wb') as f:
    #     pickle.dump(pred_model, f)
    # with open(cf.MODEL_FINAL_PATH.format('bn_linear.pkl'), 'wb') as f:
    #     pickle.dump(linear_model, f)
    name = 'siag'
    version = 'full'
    """Load model"""
    pred_model = pickle.load(open(cf.MODEL_FINAL_PATH.format('siag_prediction.pkl'), 'rb'))
    print("Load autoencoder model from {}".format(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size))))
    # dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format('/' + name + '/dfencoder_{}.pt'.format(version)))
    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size)))

    """Get prediction and negative index"""
    y_prediction = pred_model.predict(features.values)
    neg_index = np.where(y_prediction == 0)[0] 
    
    """Setup for genetic algorithm"""
    mask = ["real","real","real", "real", "real", "real", "real", "real", "real",
            "real", "real", "real", "real"
            ]
    x0 = features.loc[0].values
    df_store = pd.DataFrame(columns = col)
    df_store.loc[0] = x0
    
    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)    
    y_prediction = torch.Tensor(y_prediction)
    pos_z, neg_z = get_pos_neg_latent(y_prediction, z_representation)

    ORG_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/final_result/result/original/{}"
    data.loc[neg_index,col + ['outcome']].to_csv(ORG_PATH.format('full_negative_siag.csv'), index = False)
    sys.exit(1)
    
    # sys.exit('haha')

    
    df_store = pd.DataFrame(columns = list())
    for i in range(n_sample):
        index_ = neg_index[i]
        x0 = features.loc[index_].values
        z0 = z_representation[index_]
        """Find prototype"""
        k_instance=  n_instance
        pos_proto, neg_proto = find_proto(z0.reshape(1,-1), pos_z, neg_z, k_instance)
        
        # sys.exit('haha')
        
        print("Define problem")
        problem = CF4(x0, 
              pred_model, 
              dfencoder_model,
              scm_model,
              reg,
              pos_proto,
              col
              )
        
        print("Define algorithm")
        algorithm = NSGA2(
                pop_size=100,
                n_offsprings=50,
                sampling=get_sampling("real_random"),
                # crossover=get_crossover("real_de", prob=1.0, eta=3.0),
                crossover=get_crossover("real_sbx"),
                mutation=get_mutation("real_pm", eta=3.0),
                eliminate_duplicates=True)
        
        print("Run optimization progress")
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 20),
                        save_history=True,
                        seed = seed,
                        verbose=True)
        

        # df_store = pd.DataFrame(columns = list(features.columns))
        
        x_valid = 0
        current_dist = 9999
        
        for j in range(len(res.X)):
            xcf = res.X[j]
            ycf = pred_model.predict(np.array(xcf).reshape(1,-1))[0]
            
            if ycf == 1:
                if pure_distance(x0, xcf) <= current_dist:
                    current_dist = pure_distance(x0, xcf)
                    x_valid = xcf
                    index_ = j
        df_cf.loc[i, :-1] = x_valid
        df_cf.loc[i, 'outcome'] = pred_model.predict_proba(np.array(x_valid).reshape(1,-1))[0][1]
        df_original.loc[i, :-1] =  x0
        df_original.loc[i, 'outcome'] = pred_model.predict_proba(np.array(x0).reshape(1,-1))[0][1]
    
    df_original.to_csv(cf.FINAL_RESULT_PATH.format("siag/" + mode + "/original.csv"),
                             index = False)
    df_cf.to_csv(cf.FINAL_RESULT_PATH.format("siag/"+ mode + "/mobj-ninstance-{}-nsample-{}-size-{}-seed-{}.csv".format(n_instance, n_sample, emb_size, seed)),
                             index = False)

