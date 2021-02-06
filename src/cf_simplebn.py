import numpy as np
import pandas as pd


import torch
import torch.nn as nn

import utils
import dfencoder
import source_code

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
from pymoo.model.problem import Problem
from pymoo.util.normalization import normalize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.util import plotting

from distance import Distance
import autograd.numpy as anp


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class CF_SimpleBN(Problem):
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
                         xl=np.array([0, 0, 0]),
                         xu=np.array([1, 1, 1]),

                         **kwargs)

        self.x0 = x0
        self.pred_model = pred_model
        self.dfencoder_model = dfencoder_model
        self.scm_model = scm_model
        self.col = col
        self.proto = proto

    def _evaluate(self, xcf, out, *args, **kwargs):
        xcf[2] = self.scm_model.predict(xcf[:2].reshape(1, -1))

        """Create the distance object"""
        dist = Distance()

        pur_dist = dist.pure_distance(self.x0, xcf)
        yloss = dist.cross_entropy(self.pred_model, xcf)
        # closs = causal_loss(self.scm_model, xcf, self.x0)

        df_store = pd.DataFrame(columns=self.col)
        df_store.loc[0] = xcf
        zcf = self.dfencoder_model.get_representation(df_store)[0]

        ploss = dist.proto_loss(zcf, self.proto)

        out["F"] = anp.column_stack([yloss, pur_dist, ploss])
        # out["F"] = anp.column_stack([yloss, dist, closs, ploss])


