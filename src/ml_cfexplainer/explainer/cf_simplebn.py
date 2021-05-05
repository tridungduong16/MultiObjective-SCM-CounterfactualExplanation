import numpy as np
import pandas as pd

import torch
from pymoo.model.problem import Problem

from .distance import Distance
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
                 con_index = None,
                 cat_index = None,
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
        self.con_index = con_index

    def _evaluate(self, xcf, out, *args, **kwargs):
        xcf[2] = self.scm_model.predict(xcf[:2].reshape(1, -1))

        """Create the distance object"""
        dist = Distance(self.x0, xcf, self.pred_model, self.dfencoder_model, con_index=self.con_index)
        
        condist = dist.continous_dist()
        yloss = dist.logloss()
        # closs = causal_loss(self.scm_model, xcf, self.x0)

        df_store = pd.DataFrame(columns=self.col)
        df_store.loc[0] = xcf
        zcf = self.dfencoder_model.get_representation(df_store)[0]
        
        out["F"] = anp.column_stack([yloss, condist])
        # out["G"] = 0.5 - yloss



