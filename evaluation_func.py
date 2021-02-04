#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 16:59:22 2020

@author: trduong
"""
import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import source_code.configuration_path as cf
from source_code.helpers_algo import load_encoder_data
from source_code.evaluation import con_proximity, cat_proximity

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 

import torch 
import argparse

class Evaluation:
    def __init__(self, x0, xcf):
        """
        
        :param x0: DESCRIPTION
        :type x0: TYPE
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.x0 = x0
        self.xcf = xcf

    def convert_to_label(self, df, d):
        """
        
        :param df: DESCRIPTION
        :type df: TYPE
        :param d: DESCRIPTION
        :type d: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        for c in d.categorical_feature_names:
            le = LabelEncoder()
            le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
            df.loc[:,c] = list(le.transform(df.loc[:,c].values))
        return df

    def im1_score(self, xcf, ae1, ae0):
        """
        
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :param ae1: DESCRIPTION
        :type ae1: TYPE
        :param ae0: DESCRIPTION
        :type ae0: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return np.linalg.norm(xcf-ae1) / (np.linalg.norm(xcf-ae0) + 0.005)

    def im2_score(self, ae1, ae_full, xcf):
        """
        
        :param ae1: DESCRIPTION
        :type ae1: TYPE
        :param ae_full: DESCRIPTION
        :type ae_full: TYPE
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        return np.linalg.norm(ae1-ae_full) / (np.linalg.norm(xcf, ord=1) + 0.005)

    def change_width(self, ax, new_value) :
        """
        
        :param ax: DESCRIPTION
        :type ax: TYPE
        :param new_value: DESCRIPTION
        :type new_value: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value
            patch.set_width(new_value)
            patch.set_x(patch.get_x() + diff * .5)


    def con_prox_eval(self, df_cf, df_original, df_evaluation, method_name):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_original: DESCRIPTION
        :type df_original: TYPE
        :param df_evaluation: DESCRIPTION
        :type df_evaluation: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        loc_ = len(df_evaluation)
        subtract_df = np.sqrt((df_cf - df_original)**2)
        con_prox = subtract_df.mean(axis = 0).values.mean()
        df_evaluation.loc[loc_, 'value'] = con_prox
        df_evaluation.loc[loc_, 'method'] = method_name
        df_evaluation.loc[loc_, 'metric'] = 'continuous proximity'
        return df_evaluation


    def multiple_con_prox_eval(self, df_cf, df_original, method_name):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_original: DESCRIPTION
        :type df_original: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        subtract_df = np.sqrt((df_cf - df_original)**2)
        con_prox = subtract_df.mean(axis = 1).values
        return con_prox


    def cat_prox(self, x0, xcf):
        """
        
        :param x0: DESCRIPTION
        :type x0: TYPE
        :param xcf: DESCRIPTION
        :type xcf: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        proximity = 0
        for c in range(len(x0)):
            if x0[c] == xcf[c]:
                proximity += 1
        return proximity

    def mul_cat_prox_eval(self, df_cf, df_original, method_name):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_original: DESCRIPTION
        :type df_original: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        df_evaluation = pd.DataFrame()
        loc_ = 0
        for i in range(len(df_cf)):
            loc_ +=1
            x0 = df_original.loc[i]
            xcf = df_cf.loc[i]
            df_evaluation.loc[loc_, 'value'] = self.cat_prox(x0, xcf)
            df_evaluation.loc[loc_, 'metric'] = 'categorical proximity'
            df_evaluation.loc[loc_, 'method'] = method_name
        return df_evaluation


    def cat_prox_eval(self, df_cf, df_original, df_evaluation, method_name):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_original: DESCRIPTION
        :type df_original: TYPE
        :param df_evaluation: DESCRIPTION
        :type df_evaluation: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        loc_ = len(df_evaluation)
        cprox = 0
        for i in range(len(df_cf)):
            x0 = df_original.loc[i]
            xcf = df_cf.loc[i]
            cprox += self.cat_prox(x0, xcf)
        df_evaluation.loc[loc_, 'method'] = method_name
        df_evaluation.loc[loc_, 'metric'] = 'categorical proximity'
        df_evaluation.loc[loc_, 'value'] = cprox/len(df_cf)
        return df_evaluation

    def target_valid(self, df_cf, df_evaluation, method_name, outcome):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_evaluation: DESCRIPTION
        :type df_evaluation: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :param outcome: DESCRIPTION
        :type outcome: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        loc_ = len(df_evaluation)
        n_valid = len(df_cf[df_cf[outcome] >= 0.5])

        df_evaluation.loc[loc_, 'method'] = method_name
        df_evaluation.loc[loc_, 'metric'] = 'target valid'
        df_evaluation.loc[loc_, 'value'] = n_valid/len(df_cf)
        return df_evaluation

    def IM_calculate(self, features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg):
        """
        
        :param features: DESCRIPTION
        :type features: TYPE
        :param dfencoder_model: DESCRIPTION
        :type dfencoder_model: TYPE
        :param dfencoder_model_pos: DESCRIPTION
        :type dfencoder_model_pos: TYPE
        :param dfencoder_model_neg: DESCRIPTION
        :type dfencoder_model_neg: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        z_cer = dfencoder_model.get_representation(features)
        z_cer_pos = dfencoder_model_pos.get_representation(features)
        z_cer_neg = dfencoder_model_neg.get_representation(features)

        rec_cf = dfencoder_model.decode_to_df(z_cer)
        rec_pos_cf = dfencoder_model_pos.decode_to_df(z_cer_pos)
        rec_neg_cf = dfencoder_model_neg.decode_to_df(z_cer_neg)

        im1_val = 0
        im2_val = 0

        for i in range(len(features)):
            im1_val = self.im1_score(features.loc[i], rec_pos_cf.loc[i], rec_neg_cf.loc[i])
            im2_val = self.im2_score(rec_pos_cf.loc[i], rec_cf.loc[i], features.loc[i])

        return im1_val/len(features), im2_val/len(features)

    def multi_IM_calculate(self, features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg, method_name):
        """
        
        :param features: DESCRIPTION
        :type features: TYPE
        :param dfencoder_model: DESCRIPTION
        :type dfencoder_model: TYPE
        :param dfencoder_model_pos: DESCRIPTION
        :type dfencoder_model_pos: TYPE
        :param dfencoder_model_neg: DESCRIPTION
        :type dfencoder_model_neg: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        z_cer = dfencoder_model.get_representation(features)
        z_cer_pos = dfencoder_model_pos.get_representation(features)
        z_cer_neg = dfencoder_model_neg.get_representation(features)

        rec_cf = dfencoder_model.decode_to_df(z_cer)
        rec_pos_cf = dfencoder_model_pos.decode_to_df(z_cer_pos)
        rec_neg_cf = dfencoder_model_neg.decode_to_df(z_cer_neg)

        im1_val = 0
        im2_val = 0

        df_evaluation_im1 = pd.DataFrame()
        df_evaluation_im2 = pd.DataFrame()

        loc_ = 0

        for i in range(len(features)):
            loc_ +=1
            im1_val = self.im1_score(features.loc[i], rec_pos_cf.loc[i], rec_neg_cf.loc[i])
            im2_val = self.im2_score(rec_pos_cf.loc[i], rec_cf.loc[i], features.loc[i])
            df_evaluation_im1.loc[loc_, 'value'] = im1_val
            df_evaluation_im1.loc[loc_, 'metric'] = 'IM1'
            df_evaluation_im1.loc[loc_, 'method'] = method_name

            df_evaluation_im2.loc[loc_, 'value'] = im2_val
            df_evaluation_im2.loc[loc_, 'metric'] = 'IM2'
            df_evaluation_im2.loc[loc_, 'method'] = method_name

        return df_evaluation_im1, df_evaluation_im2


    def IM_evaluation(self, df_cf, df_evaluation, method_name, im1, im2):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_evaluation: DESCRIPTION
        :type df_evaluation: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :param im1: DESCRIPTION
        :type im1: TYPE
        :param im2: DESCRIPTION
        :type im2: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        loc_ = len(df_evaluation)
        df_evaluation.loc[loc_, 'method'] = method_name
        df_evaluation.loc[loc_, 'metric'] = 'IM1'
        df_evaluation.loc[loc_, 'value'] = im1

        loc_ = len(df_evaluation)
        df_evaluation.loc[loc_, 'method'] = method_name
        df_evaluation.loc[loc_, 'metric'] = 'IM2'
        df_evaluation.loc[loc_, 'value'] = im2

        return df_evaluation

    def causal_validity(self, df, dataset):
        """
        
        :param df: DESCRIPTION
        :type df: TYPE
        :param dataset: DESCRIPTION
        :type dataset: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        
        c_validity = 0
        for i in range(len(df)):
            # if all(x  >= 0 for x in df.loc[i]) or all(x < 0 for x in df.loc[i]):
            if all(x <= 0.1 for x in df.loc[i]):
                c_validity += 1
        return c_validity / len(df)

    def cau_val_evaluation(self, df_cf, df_original, df_evaluation, col, method_name, dataset):
        """
        
        :param df_cf: DESCRIPTION
        :type df_cf: TYPE
        :param df_original: DESCRIPTION
        :type df_original: TYPE
        :param df_evaluation: DESCRIPTION
        :type df_evaluation: TYPE
        :param col: DESCRIPTION
        :type col: TYPE
        :param method_name: DESCRIPTION
        :type method_name: TYPE
        :param dataset: DESCRIPTION
        :type dataset: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        c_validity = 0

        if (dataset == 'siag') or (dataset == 'adult'):
            df_sub = df_cf[col] - df_original[col]
            for i in range(len(df_sub)):
                if all(x  >= 0 for x in df_sub.loc[i]) or all(x < 0 for x in df_sub.loc[i]):
                    c_validity += 1
        else:
            df_sub = np.abs(df_cf[col] - df_original[col])
            for i in range(len(df_sub)):
                if all(x <= 0.1 for x in df_sub.loc[i]):
                    c_validity += 1
        c_eval = c_validity / len(df_cf)
        # c_eval = causal_validity(df_sub, dataset)
        loc_ = len(df_evaluation)
        df_evaluation.loc[loc_, 'method'] = method_name
        df_evaluation.loc[loc_, 'metric'] = 'causal validity'
        df_evaluation.loc[loc_, 'value'] = c_eval

        return df_evaluation

    def map_ordered_education(self, df):
        """
        
        :param df: DESCRIPTION
        :type df: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        c = 'education'
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        map_back = {v: k for k, v in mapping.items()}
        # print(mapping)
        education_score= {'HS-grad': 0,
                'School': 0,
                'Bachelors': 1,
                'Assoc': 1,
                'Some-college': 1,
                'Masters': 2,
                'Prof-school': 2,
                'Doctorate': 3}



        df['new_education'] = df['education'].map(map_back)
        df['new_education'] = df['new_education'].map(education_score)
        return df


    
    
 

    