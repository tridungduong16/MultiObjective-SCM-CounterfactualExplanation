#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import yaml
import logging
import sys

from ml_cfexplainer.explainer.prototype import get_pos_neg_latent
from ml_cfexplainer.explainer.prototype import find_proto
from ml_cfexplainer.explainer.cf_credit import CF_Credit

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

from scipy.spatial.distance import cdist


def find_best_solution(resX, x0, pred_model):
    
    
    y_prediction = pred_model.predict(resX)
    pos_index = np.where(y_prediction == 1)[0]
    filtered_arr  = resX[pos_index]
    if len(filtered_arr) == 0:
        return resX[0]
    distance = cdist(x0.reshape(1,-1), filtered_arr, 'euclidean')[0]
    min_index = np.where(distance == min(distance))[0]
    return filtered_arr[min_index]

if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='different_instance')
    parser.add_argument('--n_instance', type=int, default=10)
    parser.add_argument('--n_sample', type=int, default=2)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    mode = args.mode
    n_instance = args.n_instance
    n_sample = args.n_sample
    emb_size = args.emb_size
    seed = args.seed
    torch.manual_seed(seed)

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
    
    """Define features value"""

    outcome = 'default payment next month'
    features_col = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    category = ['SEX', 'EDUCATION', 'MARRIAGE']
    categorical_index = {'EDUCATION': 2, 'MARRIAGE': 3}
    cat_index = [1,2,3]
    con_index = [0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
    
    features = data[features_col]
    n_var = len(features_col)
    
    """Find range value"""
    education = list(features['EDUCATION'].unique())
    marriage = list(features['MARRIAGE'].unique())
    sex = list(features['SEX'].unique())

    """Store result"""
    logger.debug("Store result")
    df_original = pd.DataFrame(columns=list(features.columns))
    df_cf = pd.DataFrame(columns=list(features.columns))

    """Load model"""
    version = 'full'
    name = 'credit'
    pred_model = pickle.load(open(conf['prediction_model_credit'], 'rb'))
    dfencoder_model = torch.load(conf['autoencoder_model_credit'].format(version, emb_size))

    """Get prediction and negative index"""
    y_prediction = pred_model.predict(features.values)
    neg_index = np.where(y_prediction == 0)[0]

    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)
    y_prediction = torch.Tensor(y_prediction)
    pos_z, neg_z = get_pos_neg_latent(y_prediction, z_representation)

    """Setup for genetic algorithm"""
    mask = ['real']*len(features.columns)
    for i in cat_index:
        mask[i] = 'int'
        
    xu = [0]*n_var
    xl = [1]*n_var
    
    for i in cat_index:
        xu[i] = max(features.iloc[:, i])
        xl[i] = min(features.iloc[:, i])
    
    for j in con_index:
        xu[j] = max(features.iloc[:, j])
        xl[j] = min(features.iloc[:, j])        
    

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
    
    # logger.debug("Min and max")
    # logger.debug(mask)
    # logger.debug(xu)
    # logger.debug(xl)
    # logger.debug(len(xu))
    # logger.debug(np.array(xu) - np.array(xl))


    # sys.exit(1)

    if n_sample == -1:
        n_sample = len(neg_index)

    logger.debug("Number of samples {}".format(n_sample))

    # logger.debug(n_sample)
    # sys.exit(1)

    for i in range(n_sample):
        logger.debug("Sample {}".format(i))

        """Explainer"""
        index_ = neg_index[i]
        x0 = features.loc[index_].values
        # logger.debug("Original")

        # logger.debug(x0)
        # logger.debug(len(x0))


        z0 = z_representation[index_]
        k_instance = n_instance

        pos_proto, neg_proto = find_proto(z0.reshape(1, -1), pos_z, neg_z, k_instance)

        problem = CF_Credit(x0,
                 pred_model,
                 dfencoder_model,
                 features_col,
                 xl,
                 xu,
                 n_obj = 3,
                 n_var = n_var,
                 proto = pos_proto,
                 scm_model = None,
                 con_index = con_index,
                 cat_index = cat_index,
                 dict_cat_index = categorical_index)
        


        logger.debug("Define algorithm")
        algorithm = NSGA2(
            pop_size=200,
            n_offsprings=50,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True)

        logger.debug("Run optimization progress")
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 100),
                       save_history=True,
                       seed=seed,
                       verbose=True)



        df_original.loc[i,:] = x0
        df_cf.loc[i,:] =  find_best_solution(res.X, x0, pred_model)

        del problem
        del algorithm
        del res




    df_original['y'] = pred_model.predict_proba(df_original[features_col].values)[:,1]
    df_cf['y'] =  pred_model.predict_proba(df_cf[features_col].values)[:,1]

    df_original.to_csv(conf['original_credit'].format(n_sample), index = False)
    df_cf.to_csv(conf['result_credit'].format(n_instance, n_sample, emb_size, seed), index = False)



