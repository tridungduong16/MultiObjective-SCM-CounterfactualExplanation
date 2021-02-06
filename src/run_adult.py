"""
All the views for our todos application
Currently we support the following 3 views:

1. **Home** - The main view for Todos
2. **Delete** - called to delete a todo
3. **Add** - called to add a new todo

@author: trduong

"""

import sys
# MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
# sys.path.insert(1, MAIN_PATH)

import numpy as np
import pandas as pd
import copy
import logging 
import argparse
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import torch 
import torch.nn as nn

from . import utils
from . import dfencoder
from . import source_code

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader

from source_code.prototype import find_proto, get_pos_neg_latent
from dfencoder.autoencoder import AutoEncoder
from source_code import configuration_path as cf

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report

import autograd.numpy as anp

from pymoo.model.problem import Problem
from pymoo.util.normalization import normalize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.util import plotting

from .cf_adult import CF_Adult

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)  


if __name__ == "__main__":
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
    
    torch.manual_seed(seed)   
    
    
    """Load data and dataloader and normalize data"""
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 
              'continuous_features':['age','hours_per_week'], 
              'outcome_name':'income'}
    d = DataLoader(params)
    df = d.data_df
    df = d.normalize_data(df)
    
    sys.exit(1)
    """Preprocess data"""
    encoded_data = d.one_hot_encoded_data
    encoded_data = d.normalize_data(encoded_data)
    encoded_features = encoded_data.drop(columns = ['income'])
    encoded_features_tensor = torch.from_numpy(encoded_features.values).float().to(device)
    features = df.drop(columns = ['income'])

    for v in d.categorical_feature_names:
        features[v] = pd.Categorical(features[v].values)
    
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        features[c] = le.fit_transform(features[c].values)

    col = ['age','hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'race',
       'gender']
    
    features = features[col]
    """Load prediction and auto encoder model"""    
    name = 'adult'
    version = 'full'
    # dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format('/' + name + '/dfencoder_{}.pt'.format(version)))
    print("Load autoencoder model from {}".format(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size))))
    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size)))

    pred_model = torch.load('/data/trduong/DiCE/dice_ml/utils/sample_trained_models/adult.pth')
    pred_model.to(device)
    
    """Store the result"""
    df_store = pd.DataFrame(columns = list(features.columns))

    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)

    prediction_score = pred_model(encoded_features_tensor)
    prediction = torch.ge(prediction_score, 0.5).int()
    prediction = prediction.reshape(-1)
    
    neg_index = (prediction == 0).nonzero()
    
    pos_z, neg_z = get_pos_neg_latent(prediction, z_representation)
    
    """Get category constraints"""    
    arr = []
    for i in d.categorical_feature_indexes:
        arr.append(features.iloc[:,i].nunique())
        

    
    list_ = neg_index.detach().cpu().numpy().reshape(-1)

    ORG_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/final_result/result/original/{}"
    neg_features = features.iloc[list_,:]
    neg_features['income'] = 0
    neg_features.to_csv(ORG_PATH.format('full_negative_adult.csv'), index = False)
    # sys.exit(1)
    

    
    """Set up logging"""
    file_name = "multi_objective_{}.log".format(version)
    PATH = cf.LOGGING_PATH.format(file_name)
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=PATH)
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    
    loc_= 0 
    # n_sample = n_sample

    df_original = pd.DataFrame(columns = list(features.columns))
    df_counterfactual = pd.DataFrame(columns = list(features.columns))
    df_original['income'] = -1
    df_counterfactual['income'] = -1
    
    count_ = 0
    for index_ in list_[:n_sample]:
        count_ +=  1
        # df_original = pd.DataFrame(columns = list(features.columns))
        # df_counterfactual = pd.DataFrame(columns = list(features.columns))
    
        """Global variable"""    
        x_valid = 0
        best_fitness = 0
        best_prediction_score = 0
        pureDist = 10000
        
        df_store = pd.DataFrame(columns = list(features.columns))

        logger.debug("Sample index {}".format(count_))
        
        mask = ["real","real","int", "int", "int", "int", "int", "int"]
        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "int": get_sampling("int_random"),
        })
        
        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })
        
        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm", eta=3.0),
            "int": get_mutation("int_pm", eta=3.0)
        })

        """Get original instance"""
        query = features.iloc[index_,:].to_dict()
        x0 = list(query.values())
        df_store.loc[0] = x0
        z = dfencoder_model.get_representation(df_store)
        z0 = z[0]

        
        # sys.exit("Age less than 18")     
        
        """Find prototype"""
        k_instance = n_instance
        pos_proto, neg_proto = find_proto(z0.reshape(1,-1), pos_z, neg_z, k_instance)
        
        # problem = CF2(x0,
        #               d,
        #               pred_model,
        #               dfencoder_model,
        #               pos_proto,
        #               x0[6],
        #               x0[7]
        #               )

        problem = CF_Adult(x0, d, pred_model, dfencoder_model, pos_proto,x0[6],x0[7])
        
        algorithm = NSGA2(
                pop_size=100,
                n_offsprings=50,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True)
        
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 30),
                        save_history=True,
                        seed = seed,
                        verbose=True)
    

        x_valid = 0
        y_valid = -1
        current_dist = -9999
        index_ = -1
        df_store = pd.DataFrame(columns = list(features.columns))
    
    
        for i in range(len(res.X)):
            xcf = res.X[i]
            term_xcf = convert_label_to_ohe(xcf, d)
            tensor_cf = torch.from_numpy(term_xcf).float().to(device)
            ycf_pred = pred_model(tensor_cf)
            if ycf_pred >= 0.5:
                y_valid = ycf_pred.detach().cpu().numpy()
                if pure_distance(x0, xcf) >= current_dist:
                    current_dist = pure_distance(x0, xcf)
                    x_valid = xcf
                    index_ = i
        df_counterfactual.at[loc_, :-1] = x_valid
        if type(y_valid) != int:
            df_counterfactual.at[loc_, 'income'] = y_valid[0]
        else:
            df_counterfactual.at[loc_, 'income'] = y_valid
            
        df_original.at[loc_, :-1] = x0
        df_original.at[loc_, 'income'] = 0
        loc_ += 1
        print(df_original)
        print(df_counterfactual)
        df_counterfactual.to_csv( cf.FINAL_RESULT_PATH.format("adult/"+mode +"/mobj-ninstance-{}-nsample-{}-size-{}.csv".format(n_instance, n_sample, emb_size)),
                              index = False)
        # break

    print("Output file to ", cf.FINAL_RESULT_PATH.format("adult/"+ mode + "/mobj-ninstance-{}-nsample-{}-size-{}.csv".format(n_instance, n_sample, emb_size)))
    # print("Output file to ", cf.FINAL_RESULT_PATH.format("adult/original.csv"))    
    
    # df_original.to_csv(cf.FINAL_RESULT_PATH.format("adult/original.csv"),
    #                          index = False)    
    df_counterfactual.to_csv( cf.FINAL_RESULT_PATH.format("adult/"+mode +"/mobj-ninstance-{}-nsample-{}-size-{}-seed-{}.csv".format(n_instance, n_sample, emb_size, seed)),
                              index = False)

