#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:12:10 2020

@author: trduong
"""

import sys
sys.path.insert(1, '/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm')

from source_code.helpers_algo import load_encoder_data
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
import lightgbm as lgb
from sklearn.metrics import classification_report


if __name__ == '__main__':
    d, data = load_encoder_data()
    df = d.data_df
    
    education_score = {'HS-grad': 0,
        'School': 0,
        'Bachelors': 1,
        'Assoc': 1,
        'Some-college': 1,
        'Masters': 2,
        'Prof-school': 2,
        'Doctorate': 2}
    
    df['education'] = df['education'].map(education_score)

    for v in d.categorical_feature_names:
        le = preprocessing.LabelEncoder()
        df[v] = le.fit_transform(df[v])
    
    scaler = StandardScaler()
    df['age'] = scaler.fit_transform(df['age'].values.reshape(-1,1))
    
    features = df[['age', 'gender', 'race']]
    features = data[['age', 'gender_Female', 'gender_Male', 
                     'race_Other', 'race_White'
        #               'occupation_Other/Unknown', 'occupation_Professional',
        # 'occupation_Sales', 'occupation_Service', 'occupation_White-Collar'
                     
                     ]]

    label = df['education']
    
    
    d_train=lgb.Dataset(features, label=label)
    #Specifying the parameter
    # params={}
    # params['learning_rate']=0.03
    # params['boosting_type']='gbdt' #GradientBoostingDecisionTree
    # params['objective']='multiclass' #Binary target feature
    # params['metric']='multi_logloss' #metric for binary classification
    # params['max_depth']=10
    # params['num_class']=4 #no.of unique values in the target class not inclusive of the end value
    # params['is_unbalance']= True
    # params['scale_pos_weight']= 99
    #train the model 
    
    params = {
        'learning_rate': 0.15,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 20,  # 2^max_depth - 1
        'max_depth': 20,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'is_unbalance':'true', # because training data is extremely unbalanced 
        # 'objective' : 'binary'
        'objective' : 'multiclass',
        'num_class': 3

    }
    
    
    clf=lgb.train(params,d_train,100) #train the model on 100 epocs
    y_pred=clf.predict(features)
    prediction = [np.argmax(line) for line in y_pred]

    # clf = DecisionTreeClassifier(max_depth=50, min_samples_leaf = 10, random_state=0)
    # clf = GradientBoostingClassifier(learning_rate=0.1, 
    #                                  n_estimators=60,
    #                                  max_depth=9, 
    #                                  min_samples_split=1200, 
    #                                  min_samples_leaf=60, 
    #                                  subsample=0.8, 
    #                                  random_state=10)
    # clf.fit(features, label)
    # prediction = clf.predict(features)
    

    y_test = df['education'].values
    y_pred = prediction
    labels = [0, 1, 2] 

    print(classification_report(y_test, y_pred, labels)) #classification report from sklearn

    
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    plt.imshow(cnf_matrix, cmap=plt.cm.Blues) #plot confusion matrix grid
    threshold = cnf_matrix.max() / 2 #threshold to define text color
    for i in range(cnf_matrix.shape[0]): #print text in grid
        for j in range(cnf_matrix.shape[1]): 
            plt.text(j, i, cnf_matrix[i,j], color="w" if cnf_matrix[i,j] > threshold else 'black')
    tick_marks = np.arange(len(labels)) #define labeling spacing based on number of classes
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.tight_layout()


    
    