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
from pymoo.model import problem
import autograd.numpy as np
from pymoo.model.problem import Problem
from pymoo.util.normalization import normalize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.util import plotting
import autograd.numpy as anp

from distance import Distance


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class CF_Sangiovese(Problem):
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
                         xl=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                         xu=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                         **kwargs)

        self.x0 = x0
        self.pred_model = pred_model
        self.dfencoder_model = dfencoder_model
        self.scm_model = scm_model
        self.reg = reg
        self.col = col
        self.proto = proto

    def _evaluate(self, xcf, out, *args, **kwargs):
        """Create the distance object"""
        dist = Distance()

        mse = dist.mean_squared_error(self.x0, xcf)
        yloss = dist.cross_entropy(self.pred_model, self.xcf)
        # closs = structural_causal_equation('BunchN', scm_model, xcf, x0, col)
        key = 'BunchN'
        closs = dist.scm_loss(self.reg, self.x0, self.xcf, key, self.scm_model, self.col)

        df_store = pd.DataFrame(columns=self.col)
        df_store.loc[0] = self.xcf
        zcf = self.dfencoder_model.get_representation(df_store)[0]

        ploss = dist.proto_loss(zcf, self.proto)

        out["F"] = anp.column_stack([yloss, mse, closs, ploss])



