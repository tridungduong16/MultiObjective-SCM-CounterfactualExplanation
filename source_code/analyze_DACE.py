# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import sys
import pandas as pd

sys.path.insert(1, '/home/trduong/Data/DiCE')

import dice_ml
from dice_ml.utils import helpers # helper functions

dataset = helpers.load_adult_income_dataset()
d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')

backend = 'PYT'
ML_modelpath = "/home/trduong/Data/DiCE/dice_ml/utils/sample_trained_models/adult.pth"

m = dice_ml.Model(model_path= ML_modelpath, backend=backend)

exp = dice_ml.Dice(d, m)

# query instance in the form of a dictionary; keys: feature name, values: feature value
query_instance = {'age':22, 
                  'workclass':'Private', 
                  'education':'HS-grad', 
                  'marital_status':'Single', 
                  'occupation':'Service',
                  'race': 'White', 
                  'gender':'Female', 
                  'hours_per_week': 45}


dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=1, desired_class="opposite")


