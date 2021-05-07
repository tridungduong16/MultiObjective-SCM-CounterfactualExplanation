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
from ml_cfexplainer.explainer.cf_simplebn import CF_SimpleBN

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation

from scipy.spatial.distance import cdist


def find_best_solution(resX, x0, pred_model):
    y_prediction = pred_model.predict(resX)
    pos_index = np.where(y_prediction == 1)[0]
    filtered_arr  = resX[pos_index]

    distance = cdist(x0.reshape(1,-1), filtered_arr, 'euclidean')[0]
    min_index = np.where(distance == min(distance))[0]

    return filtered_arr[min_index]

if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='different_instance')
    parser.add_argument('--n_instance', type=int, default=10)
    parser.add_argument('--n_sample', type=int, default=-1)
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
    file_handler = logging.FileHandler(filename=conf['log_path'].format('simpleBN.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)

    """Load data and dataloader and normalize data"""
    logger.debug("Load data and dataloader and normalize data")
    features_col = ['x1', 'x2', 'x3']
    con_index = [0,1,2]
    data = pd.read_csv(conf['processed_simple_bn'])
    features = data[features_col]

    """Store result"""
    logger.debug("Store result")
    df_original = pd.DataFrame(columns=list(data.columns))
    df_cf = pd.DataFrame(columns=list(data.columns))

    """Load model"""
    version = 'full'
    name = 'simple_bn'
    pred_model = pickle.load(open(conf['prediction_model_simple_bn'], 'rb'))
    scm_model = pickle.load(open(conf['scm_model_simple_bn'], 'rb'))
    dfencoder_model = torch.load(conf['model_path_simple_bn'].format('dfencoder_{}_{}.pt'.format(version, emb_size)))

    """Get prediction and negative index"""
    y_prediction = pred_model.predict(features.values)
    neg_index = np.where(y_prediction == 0)[0]

    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)
    y_prediction = torch.Tensor(y_prediction)
    pos_z, neg_z = get_pos_neg_latent(y_prediction, z_representation)

    """Setup for genetic algorithm"""
    logger.debug("Setup for genetic algorithm")
    mask = ["real", "real", "real"]

    if n_sample == -1:
        n_sample = len(neg_index)

    for i in range(n_sample):

        """Explainer"""
        index_ = neg_index[i]
        x0 = features.loc[index_].values
        z0 = z_representation[index_]
        k_instance = n_instance

        pos_proto, neg_proto = find_proto(z0.reshape(1, -1), pos_z, neg_z, k_instance)

        problem = CF_SimpleBN(x0,
                     pred_model,
                     dfencoder_model,
                     scm_model,
                     pos_proto,
                     features_col,
                     con_index)

        logger.debug("Define algorithm")
        algorithm = NSGA2(
            pop_size=200,
            n_offsprings=50,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=1.0, eta=3.0),
            mutation=get_mutation("real_pm", eta=3.0),
            eliminate_duplicates=True)

        logger.debug("Run optimization progress")
        res = minimize(problem,
                       algorithm,
                       ('n_gen', 1),
                       save_history=True,
                       seed=seed,
                       verbose=True)


        df_original.at[i, :-1] = x0
        df_cf.at[i, :-1] =  find_best_solution(res.X, x0, pred_model)

        del problem
        del algorithm
        del res




    df_original['y'] = pred_model.predict_proba(df_original[features_col].values)[:,1]
    df_cf['y'] =  pred_model.predict_proba(df_cf[features_col].values)[:,1]

    print("Output file to ", conf['result_simple_bn'].format(n_instance, n_sample, emb_size, seed))
    df_original.to_csv(conf['original_simple_bn'].format(n_sample), index = False)
    df_cf.to_csv(conf['result_simple_bn'].format(n_instance, n_sample, emb_size, seed), index = False)



