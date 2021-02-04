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

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class CF2(Problem):
    def __init__(self, x0,
                 d,
                 pred_model,
                 dfencoder_model,
                 pos_proto,
                 features,
                 x6,
                 x7,
                 n_var=8,
                 **kwargs):
        super().__init__(n_var=n_var,
                         n_obj=4,
                         n_constr=1,
                         xl=np.array([0, 0, 0, 0, 0, 0, x6, x7]),
                         xu=np.array([0.95, 0.85, 3, 7, 4, 5, x6, x7]),
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         **kwargs)
        self.x0 = x0
        self.pred_model = pred_model
        self.proto = pos_proto
        self.dfencoder_model = dfencoder_model
        self.features = features
    def _evaluate(self, xcf, out, *args, **kwargs):
        df_store = pd.DataFrame(columns=list(self.features.columns))

        df_store.loc[0] = self.x0
        df_store.loc[1] = self.xcf
        """Get the representation"""
        z = self.dfencoder_model.get_representation(df_store)
        zcf = z[1]

        """Compute prediction loss"""
        # ycf = torch.Tensor([1.0]).to(device)
        # yloss = compute_yloss(xcf, ycf, self.pred_model, self.d)
        # con_dist = continous_dist(self.x0, xcf, self.d)
        # cat_dist = cat_representation_dist(self.dfencoder_model, self.x0, xcf)
        #
        # """Compute prototype loss"""
        # ploss = proto_loss(zcf, self.proto)
        #
        # """Constraints"""
        # g1 = self.x0[0] - self.xcf[0]
        #
        # out["F"] = anp.column_stack([yloss, con_dist, cat_dist, ploss])
        # out["F"] = anp.column_stack([yloss, con_dist, cat_dist])
        out["G"] = np.column_stack([g1])



