# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

import torch 
import torch.nn as nn

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader
from source_code.prototype import find_proto, get_pos_neg_latent
from dfencoder.autoencoder import AutoEncoder
import source_code.configuration_path as cf
import pickle





"""Pure distance"""
def pure_distance(x0, xcf):
    return mean_squared_error(x0, xcf)

"""Compute prediction loss"""
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



"""Compute the median absolute deviation in training data"""
def MAD(df, name):  
    if name == 'simple_bn':
        col = ['x1', 'x2', 'x3']
    elif name == 'siag':
        col = ['SproutN', 'BunchN', 'WoodW', 'SPAD06','NDVI06', 'SPAD08', 'NDVI08', 
               'Acid', 'Potass', 'Brix', 'pH', 'Anthoc', 'Polyph']
    
    mad = []
    
    for c in col:
        m = np.mean(df[c].values - df[c].median())
        mad.append(m)

    return mad 

"""Compute loss for MAD"""
def distance_mad(x0, xcf, df, name):
    """MAD features importance"""
    mad = np.array(MAD(df, name))   
    
    """distance"""
    distance = (x0-xcf)/mad
    
    return np.mean(distance) 

"""Compute loss for CERTIFAI"""
# def certifai_distance(x0, xcf):
#     cat_indexes = [1,2,3,4,5,6]
#     con_indexes = [0,7]
#     loss = 0
#     for i in cat_indexes:
#         if x0[i] == xcf[i]:
#             loss -= 1
#     for j in con_indexes:
#         loss += np.sqrt((xcf[j] - x0[j])**2)
    
#     return loss 


if __name__ == "__main__":
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev) 
    
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='simple_bn')
    parser.add_argument('--distance', type=str, default='certifai')
    parser.add_argument('--n_instance', type=int, default=5)

    args = parser.parse_args()
    name = args.name
    distance = args.distance
    n_instance = args.n_instance
    
        
    # name = 'simple_bn'
    # name = 'siag'
    # distance = 'mad'
    # distance = 'certifai'
    
    print("Dataset {}".format(name))
    seed = 0
    torch.manual_seed(seed)   
 
    """Load dataset """
    DATA_PATH = cf.BASE_PATH.format('cf-feasibility/generativecf/data/data-generativecf/{}')
    
    if name == 'simple_bn':
        SIMPLE_BN = DATA_PATH.format('bn1.csv')
        data = pd.read_csv(SIMPLE_BN)
        data = data[['x1', 'x2', 'x3', 'y']]
        data = data[['x1', 'x2', 'x3', 'y']]
        data['x1'] = (data['x1']-data['x1'].min())/(data['x1'].max()-data['x1'].min())
        data['x2'] = (data['x2']-data['x2'].min())/(data['x2'].max()-data['x2'].min())
        data['x3'] = (data['x3']-data['x3'].min())/(data['x3'].max()-data['x3'].min())
        features = data[['x1', 'x2', 'x3']]

        """Load model"""
        pred_model = pickle.load(open(cf.MODEL_FINAL_PATH.format('bn_prediction.pkl'), 'rb'))
        linear_model = pickle.load(open(cf.MODEL_FINAL_PATH.format('bn_linear.pkl'), 'rb'))
        dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format('dfencoder_bn_full.pt'))
    
    elif name == 'siag':
        SIMPLE_BN = DATA_PATH.format('sangiovese.csv')
        data = pd.read_csv(SIMPLE_BN)
        col = ['SproutN', 'BunchN', 'WoodW', 'SPAD06',
       'NDVI06', 'SPAD08', 'NDVI08', 'Acid', 'Potass', 'Brix', 'pH', 'Anthoc',
       'Polyph']
        data.loc[data['GrapeW'] > 0, 'outcome'] = 1
        data.loc[data['GrapeW'] <= 0, 'outcome'] = 0
        features = data[col]
        
        """Load model"""
        pred_model = pickle.load(open(cf.MODEL_FINAL_PATH.format('siag_prediction.pkl'), 'rb'))
        dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format('dfencoder_siag_full.pt'))
    
    
    """Store the result"""
    df_store = pd.DataFrame(columns = list(features.columns))
       
    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)

    """Get prediction and negative index"""
    y_prediction = pred_model.predict(features.values)
    neg_index = np.where(y_prediction == 0)[0] 
    
    
    """Parameter setup"""
    threshold = -0.5
    num_generations, num_parents_mating, sol_per_pop =  cf.num_generations, cf.num_parents_mating, cf.sol_per_pop
    init_range_low, init_range_high = cf.init_range_low, cf.init_range_high
    parent_selection_type, keep_parents = cf.parent_selection_type, cf.keep_parents
    crossover_type, mutation_type, mutation_percent_genes = cf.crossover_type, cf.mutation_type,cf.mutation_percent_genes
    mutation_probability = cf.mutation_probability
    
    last_fitness = 0
        
    df_original = pd.DataFrame(columns = list(features.columns))
    df_counterfactual = pd.DataFrame(columns = list(features.columns))
    
    if name == 'simple_bn':
        y_name = 'y'
    elif name == 'siag':
        y_name = 'outcome'
        
    df_original[y_name] = -1
    df_counterfactual[y_name] = -1

    file_name = "genetic_version_{}.log".format(name)
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
    
    n_sample = 10
    for i in range(n_sample):
        index_ = neg_index[i]
    # for index_ in neg_index:
        """Global variable"""    
        x0 = features.loc[index_].values
        length = len(x0)
        x_valid = np.array([0]*length)
        best_fitness = 0
        best_prediction_score = 0
        pureDist = 10000
        
        df_store = pd.DataFrame(columns = list(features.columns))
        
        
        logger.debug("Sample index {}".format(i))
        
        """Number of genes"""
        num_genes = len(x0)
        
        """Fitness function"""
        def fitness_func(xcf, solution_idx):
            df_store.loc[0] = x0
            df_store.loc[1] = xcf   
            
            
            xcf[0] = max(xcf[0],0.1)
            xcf[0] = min(xcf[0],0.9)
            xcf[1] = max(xcf[1],0.1)
            xcf[1] = min(xcf[1],0.9)
            xcf[2] = max(xcf[2],0.1)
            xcf[2] = min(xcf[2],0.9)                        
            """Compute prediction loss"""
            yloss = cross_entropy(pred_model, xcf, targets=1,epsilon=1e-12)
            
            if distance == 'certifai':
                """Certifai distance"""
                dist = pure_distance(x0, xcf)
            elif distance == 'mad':
                """MAD distance"""
                dist = distance_mad(x0, x_valid, features, name)
            
            """Compute final loss"""
            final_loss = 4*yloss + 6*dist
            
            return -final_loss 
        
        
        """Callback function"""
        def callback_generation(ga_instance):
            global x_valid
            global pureDist
            global last_fitness

            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            
            """Logging"""
            fitness = ga_instance.best_solution()[1]
            change = ga_instance.best_solution()[1] - last_fitness
            logger.debug("Generation = {generation}".format(generation=ga_instance.generations_completed))
            logger.debug("Fitness {:.4f} and Change {:.4f}".format(fitness,change))
            last_fitness = ga_instance.best_solution()[1]            
            predictions = pred_model.predict(np.array(solution).reshape(1,-1))
            p_dist = pure_distance(x0, solution)
            
            # print("Solution ", solution)
            # print("Prediction ", predictions)
            logger.debug("Current solution {}".format(x_valid))
            logger.debug("-------------------------------------------")
            
            if predictions == 1:
                # print("Prediction is valid")
                # print("Current distance {} and expected distance {}".format(pureDist, p_dist))
                if pureDist >= p_dist:     
                    # print("Update")
                    term = copy.deepcopy(solution)
                    x_valid = term
                    pureDist = p_dist 
                
            
            
                        

        """Creating an instance of the GA class inside the ga module."""
        ga_instance = pygad.GA(
                               # initial_population = [x0]*sol_per_pop,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating, 
                               fitness_func=fitness_func,
                               sol_per_pop=sol_per_pop, 
                               num_genes=num_genes,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               callback_generation=callback_generation,
                                mutation_probability = mutation_probability,
                                init_range_low=0.0,
                                init_range_high=1.0,
                                # gene_space=[None, 
                                #             range(arr[0]), 
                                #             range(arr[1]),
                                #             range(arr[2]),
                                #             range(arr[3]),
                                #             int(x0[5]),
                                #             int(x0[6]),
                                #             None
                                #             ],
                                random_mutation_min_val=0,
                                random_mutation_max_val=1.0
                               )
        
        ga_instance.run()
        
        """Returning the details of the best solution."""
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        """Logging"""
        logger.debug("Parameters of the best solution : {solution}".format(solution=solution))
        logger.debug("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        logger.debug("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        
        
        # if type(x_valid) != int:
        #     logger.debug("Found the valid solution")
        #     logger.debug(x_valid)
        # solution = x_valid
        # else:
        #     logger.debug("No solution")

        
        df_store.loc[1] = x_valid
        # logger.debug("Final result")
        # logger.debug(df_store)

        df_original.loc[i, :-1] = x0
        df_counterfactual.loc[i, :-1] = x_valid 
        
        y0 = pred_model.predict_proba(np.array(x0).reshape(1,-1))[0][1]
        y1 = pred_model.predict_proba(np.array(x_valid).reshape(1,-1))[0][1]
        
        if name == 'simple_bn':
            df_original.loc[i, 'y'] = y0
            df_counterfactual.loc[i, 'y'] = y1
        else:
            df_original.loc[i, 'outcome'] = y0
            df_counterfactual.loc[i, 'outcome'] = y1
        
        logger.debug("Original")
        logger.debug(df_original)

        logger.debug("Counterfactual")
        logger.debug(df_counterfactual)
        
        logger.debug("Done!!!!!!!!!")
        logger.debug("***********************************************************************")

        del df_store
        del ga_instance
        # break 
    
    
    torch.cuda.empty_cache()

    PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/final_result/result/{}"
    df_original = df_original.round(3)
    df_counterfactual = df_counterfactual.round(3)

    logger.debug("Ouput to file............")
    logger.debug("***********************************************************************")
    logger.debug("***********************************************************************")
    logger.debug("\n")

    df_original.to_csv(PATH.format(name + "/"+distance+".csv"),
                        index = False)
    df_counterfactual.to_csv(PATH.format(name  + "/"+ distance + ".csv"),
                             index = False)
