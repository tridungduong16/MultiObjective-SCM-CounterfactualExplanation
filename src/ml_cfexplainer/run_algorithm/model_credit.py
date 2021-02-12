#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:19:05 2021

@author: trduong
"""

import pandas as pd
import yaml
import logging
import sys
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
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

    outcome = 'default payment next month'
    feature = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
                'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
                'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
                'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    category = ['SEX', 'EDUCATION', 'MARRIAGE']

    """Build prediction model"""
    classifier = GradientBoostingClassifier(random_state=0).fit(data[feature], data[outcome])
    with open(conf['prediction_model_credit'], 'wb') as f:
        pickle.dump(classifier, f)
    data['y_prediction'] = classifier.predict(data[feature])

    """Check the performance"""
    acc = accuracy_score(data['default payment next month'], data['y_prediction'])
    print("Accuracy score {:.4f}".format(acc))
    



