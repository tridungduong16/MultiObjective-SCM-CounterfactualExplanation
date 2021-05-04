from .distance import Distance

import numpy as np
import pandas as pd
import torch
import autograd.numpy as anp
from pymoo.model.problem import Problem



if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

"""
Number of variable: 30
Number of objective function: 2
"""


class CF_Adult(Problem):
    
    def __init__(self, x0,
                 d,
                 features,
                 pred_model,
                 dfencoder_model,
                 pos_proto,
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
        df_store.loc[1] = xcf
        """Get the representation"""
        z = self.dfencoder_model.get_representation(df_store)
        zcf = z[1]

        """Create the distance object"""
        dist = Distance()

        """Compute prediction loss"""
        ycf = torch.Tensor([1.0]).to(device)
        yloss = dist.compute_yloss(self.xcf, ycf, self.pred_model, self.d)
        con_dist = dist.continous_dist(self.x0, self.xcf, self.d)
        cat_dist = dist.cat_representation_dist(self.dfencoder_model, self.x0, self.xcf)

        """Compute prototype loss"""
        ploss = dist.proto_loss(zcf, self.proto)

        """Constraints"""
        g1 = self.x0[0] - self.xcf[0]

        out["F"] = anp.column_stack([yloss, con_dist, cat_dist, ploss])
        # out["F"] = anp.column_stack([yloss, con_dist, cat_dist])
        out["G"] = np.column_stack([g1])



