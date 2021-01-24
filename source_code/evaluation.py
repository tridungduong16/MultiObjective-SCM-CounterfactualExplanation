
import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)


import torch 
import numpy as np

from source_code.auto_encoder import autoencoder
from source_code.helpers_algo import load_encoder_data
import source_code.configuration_path as cf



seed = 0
torch.manual_seed(seed)

# def IM(zcf_org, zcf_target, zcf_full, xcf):
#     regularization = 0.1
#     im1 = torch.dist(xcf, zcf_target, 2) / (torch.dist(xcf, zcf_org, 2) + regularization)
#     im2 = torch.dist(zcf_target, zcf_full, 2) / (torch.norm(xcf, p =1) + regularization)
       
#     return im1, im2 

def con_proximity(x0, xcf, con_indexes):
    proximity = 0
    for c in con_indexes:
        proximity += np.absolute(x0[c] - xcf[c])
    proximity = np.mean(proximity)
    return proximity

def cat_proximity(x0, xcf, cat_indexes):
    proximity = 0
    for c in cat_indexes:
        if x0[c] == xcf[c]:
            proximity += 1
    return proximity 

    

