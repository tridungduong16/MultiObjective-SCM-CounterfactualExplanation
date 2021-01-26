import numpy as np
import pandas as pd
import copy
import logging
import argparse
import pickle
import matplotlib.pyplot as plt
import sys
import os

import scipy 
# sys.path.insert(0, os.path.abspath('../'))

# from scipy.spatial.distance import cosine

import torch
import torch.nn as nn

import utils
import dfencoder
import source_code

# from . import utils
# from . import dfencoder
# from . import source_code

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader

from source_code.prototype import find_proto, get_pos_neg_latent
from dfencoder.autoencoder import AutoEncoder
from source_code import configuration_path as cf

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

import pymoo 
# from pymoo.model import problem 
# import autograd.numpy as np
# from pymoo.model.problem import Problem
# from pymoo.util.normalization import normalize
# from pymoo.algorithms.nsga2 import NSGA2
# from pymoo.factory import get_problem
# from pymoo.optimize import minimize
# from pymoo.visualization.scatter import Scatter
# from pymoo.factory import get_sampling, get_crossover, get_mutation
# from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
# from pymoo.util import plotting

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class Distance:
    """ Distance class
    """

    def __init__(self, x0, xcf):
        """Constructor method
        """
        self.x0 = x0
        self.xcf = xcf
        
    def pure_distance(self):
        """
        
        :param x0: original instance
        :type x0: int
        :param xcf: counterfactual instance
        :type xcf: int
        :return: distance between two instance
        :rtype: float
    
        """
        cat_indexes = [1, 2, 3, 4]
        con_indexes = [0, 7]
    
        cat_sim = 0
        con_sim = 0
        for i in cat_indexes:
            if self.x0[i] == self.xcf[i]:
                cat_sim += 1
    
        # cat_sim /= 4
    
        for j in con_indexes:
            con_sim -= np.sqrt((self.xcf[j] - self.x0[j]) ** 2)
    
        # con_sim /= 2
    
        return cat_sim + con_sim
    
    
    def continous_dist(self):
        """
        
        :param x0: original instance
        :type x0: int
        :param xcf: counterfactual instance
        :type xcf: int
        :return: distance between two instance
        :rtype: float
    
        """
        loss = 0
        loss += np.sqrt((self.xcf[7] - self.x0[7]) ** 2)
        loss += np.sqrt((self.xcf[0] - self.x0[0]) ** 2)
        return loss
    
        
    def latent_distance(self, z0, zcf):
        """
        
        :param x0: original instance
        :type x0: int
        :param xcf: counterfactual instance
        :type xcf: int
        :return: distance between two instance
        :rtype: float
    
        """
        criterion = nn.MSELoss()
        dist = criterion(z0, zcf).detach().cpu().numpy()
        return dist
    
        
    def two_cate_dist(self, model, cat1, cat2):
        """
        
        :param x0: original instance
        :type x0: int
        :param xcf: counterfactual instance
        :type xcf: int
        :return: distance between two instance
        :rtype: float
    
        """
        feature = model.categorical_fts[self.feature]
        cats = feature['cats']
        cat1_index = cats.index(cat1)
        cat2_index = cats.index(cat2)
    
        emb = self.feature['embedding']
        emb_cat1 = emb.weight.data.cpu().numpy()[cat1_index, :]
        emb_cat2 = emb.weight.data.cpu().numpy()[cat2_index, :]
        similarity = cosine(emb_cat1, emb_cat2)
    
        return similarity
    
    
    
    
    def cat_representation_dist(self):
        """ Compute the categorical distance in latent space
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        dict_ = {'workclass': 2,
                  'education': 3,
                  'marital_status': 4,
                  'occupation': 5}
    
        similarity = 0
        for k, v in dict_.items():
            similarity += self.two_cate_dist(self.dfencoder_model, k, self.x0[v], self.xcf[v])
    
        return similarity
    
        
    def proto_loss(self, zcf, proto):
        """
        
        :param zcf: DESCRIPTION
        :type zcf: TYPE
        :param proto: DESCRIPTION
        :type proto: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        criterion = nn.MSELoss()
        proto_loss = criterion(proto.reshape(-1), zcf)
        proto_loss = proto_loss.detach().cpu().numpy()
        return proto_loss
    
    
    # """Compute the constraint loss: Age have to be larger than original instance"""
    
    
    def constraints_loss(self):
        """ Compute the constraint loss: Age have to be larger than original instance
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        criterion = nn.MarginRankingLoss(margin=0)
        loss1 = criterion(self.xcf[0].reshape(-1, 1), self.x0[0].reshape(-1, 1), torch.Tensor([1]).to(device))
        # criterion = nn.MarginRankingLoss(margin = 0)
        # loss2 = criterion(torch.Tensor([50]).to(device), x0[0].reshape(-1,1), torch.Tensor([1]).to(device))
        # criterion = nn.MarginRankingLoss(margin = 1)
        # loss3 = criterion(xcf[7].reshape(-1,1), x0[7].reshape(-1,1), torch.Tensor([1]).to(device))
    
        loss = loss1
        loss = loss.detach().cpu().numpy()
        return loss
    
        
    def compute_yloss(self, ycf, prediction_model, d):
        """ Compute the prediction loss function
        
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :param ycf: DESCRIPTION
        :type ycf: TYPE
        :param prediction_model: DESCRIPTION
        :type prediction_model: TYPE
        :param d: DESCRIPTION
        :type d: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        term_xcf = self.convert_label_to_ohe(self.xcf, d)
        tensor_cf = torch.from_numpy(term_xcf).float().to(device)
        ycf_pred = prediction_model(tensor_cf)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(ycf, ycf_pred)
        loss = loss.detach().cpu().numpy()
        return loss
    

    
    def causal_loss_adult(self):
        """
        
        :return: DESCRIPTION
        :rtype: TYPE

        """
        x0 = np.array(self.x0)
        xcf = np.array(self.xcf)
        subset_dfencoder = torch.load(cf.MODEL_FINAL_PATH.format('subset_dfencoder.pt'))
    
        dict_ = {'age': self.xcf[0],
                  'workclass': int(self.xcf[1]),
                  'education': int(self.xcf[2]),
                  'marital_status': int(self.xcf[3]),
                  'occupation': int(self.xcf[4]),
                  'race': int(self.xcf[5]),
                  'gender': int(self.xcf[6]),
                  'hours_per_week': self.xcf[7]}
    
        c = 'occupation'
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        mapping1 = {v: k for k, v in mapping.items()}
        mapping2 = {k: v for k, v in mapping.items()}
    
        dictionary_map = {
            'Blue-Collar': 1,
            'Other/Unknown': 1,
            'Professional': 2,
            'White-Collar': 2,
            'Sales': 1,
            'Service': 1
        }
    
        map_back = {
            2: 'White-Collar',
            1: 'Sales'
        }
    
        df_term = pd.DataFrame({k: [v] for k, v in dict_.items()})
        df_term = df_term[['age', 'race', 'gender', 'education', 'workclass', 'marital_status']]
    
        z = subset_dfencoder.get_representation(df_term)
        z = z.detach().cpu().numpy()
    
        # clf = pickle.loads(cf.MODEL_FINAL_PATH.format('occupation_binary_classifier.pkl'))
        clf = pickle.load(open(cf.MODEL_FINAL_PATH.format('occupation_binary_classifier.pkl'), 'rb'))
    
        prediction = clf.predict(z)
        prediction = mapping2[map_back[prediction[0]]]
    
        tensor0 = torch.from_numpy(np.array([self.x0[4]])).float().to(device)
        tensor_cf = torch.from_numpy(np.array([prediction])).float().to(device)
    
        criterion = nn.MarginRankingLoss(margin=0.01)
        loss = criterion(tensor0, tensor_cf, torch.Tensor([0]).to(device))
    
        return loss.detach().cpu().numpy()

    def causal_loss_sangio(linear_model, xcf, x0):
        x3 = linear_model.predict(xcf[:2].reshape(1, -1))
        loss = np.sqrt((x3 - x0[2]) ** 2)
        return loss

    def MAD(self, df):
        """Compute the median absolute deviation in training data
        
        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        mad_0 = np.mean(df['age'].values - df['age'].median())
        mad_7 = np.mean(df['hours_per_week'].values - df['hours_per_week'].median())
    
        mad_1 = 4
        mad_2 = 6
        mad_3 = 1
        mad_4 = 2
        mad_5 = 3
        mad_6 = 2
        return [mad_0, mad_1, mad_2, mad_3, mad_4, mad_5, mad_6, mad_7]
    
    
    
    def distance_mad(self):
        """Compute loss for MAD
        
        :param x0: DESCRIPTION
        :type x0: TYPE
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE
    
        """
        mean_absolute_deviation = np.array(MAD(df))
    
        distance = (x0 - xcf) / mean_absolute_deviation
    
        return np.mean(distance)

    def certifai_distance(self, x0, xcf):
        """Compute loss for CERTIFAI

        :param x0: DESCRIPTION
        :type x0: TYPE
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        cat_indexes = [1, 2, 3, 4, 5, 6]
        con_indexes = [0, 7]
        loss = 0
        for i in cat_indexes:
            if x0[i] == xcf[i]:
                loss -= 1
        for j in con_indexes:
            loss += (xcf[j] - x0[j])

        return loss


    def cross_entropy(self, pred_model, xcf, targets=1, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
               targets (N, k) ndarray
        Returns: scalar
        """

        predictions = pred_model.predict(np.array(xcf).reshape(1, -1))
        predictions = predictions.reshape(-1, 1)
        targets = np.array(targets).reshape(-1, 1)

        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = np.sum(targets * np.log(predictions + 1e-9)) / N
        return ce


