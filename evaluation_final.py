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

def convert_to_label(df, d):
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        df.loc[:,c] = list(le.transform(df.loc[:,c].values))
    return df 

def im1_score(xcf, ae1, ae0):
    return np.linalg.norm(xcf-ae1) / (np.linalg.norm(xcf-ae0) + 0.005)

def im2_score(ae1, ae_full, xcf):
    return np.linalg.norm(ae1-ae_full) / (np.linalg.norm(xcf, ord=1) + 0.005)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value
        patch.set_width(new_value)
        patch.set_x(patch.get_x() + diff * .5)


def con_prox_eval(df_cf, df_original, df_evaluation, method_name):
    loc_ = len(df_evaluation)
    subtract_df = np.sqrt((df_cf - df_original)**2)
    con_prox = subtract_df.mean(axis = 0).values.mean()
    df_evaluation.loc[loc_, 'value'] = con_prox     
    df_evaluation.loc[loc_, 'method'] = method_name
    df_evaluation.loc[loc_, 'metric'] = 'continuous proximity'
    return df_evaluation


def multiple_con_prox_eval(df_cf, df_original, method_name):
    subtract_df = np.sqrt((df_cf - df_original)**2)
    con_prox = subtract_df.mean(axis = 1).values
    return con_prox

    
def cat_prox(x0, xcf):
    proximity = 0
    for c in range(len(x0)):
        if x0[c] == xcf[c]:
            proximity += 1
    return proximity 

def mul_cat_prox_eval(df_cf, df_original, method_name):
    df_evaluation = pd.DataFrame()
    loc_ = 0
    for i in range(len(df_cf)):
        loc_ +=1
        x0 = df_original.loc[i]
        xcf = df_cf.loc[i]
        df_evaluation.loc[loc_, 'value'] = cat_prox(x0, xcf)
        df_evaluation.loc[loc_, 'metric'] = 'categorical proximity'
        df_evaluation.loc[loc_, 'method'] = method_name
    return df_evaluation


def cat_prox_eval(df_cf, df_original, df_evaluation, method_name):
    loc_ = len(df_evaluation)
    cprox = 0
    for i in range(len(df_cf)):
        x0 = df_original.loc[i]
        xcf = df_cf.loc[i]
        cprox += cat_prox(x0, xcf)
    df_evaluation.loc[loc_, 'method'] = method_name
    df_evaluation.loc[loc_, 'metric'] = 'categorical proximity'
    df_evaluation.loc[loc_, 'value'] = cprox/len(df_cf)
    return df_evaluation

def target_valid(df_cf, df_evaluation, method_name, outcome):
    loc_ = len(df_evaluation)
    n_valid = len(df_cf[df_cf[outcome] >= 0.5])

    df_evaluation.loc[loc_, 'method'] = method_name
    df_evaluation.loc[loc_, 'metric'] = 'target valid'
    df_evaluation.loc[loc_, 'value'] = n_valid/len(df_cf)
    return df_evaluation

def IM_calculate(features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg):
    z_cer = dfencoder_model.get_representation(features)
    z_cer_pos = dfencoder_model_pos.get_representation(features)
    z_cer_neg = dfencoder_model_neg.get_representation(features)

    rec_cf = dfencoder_model.decode_to_df(z_cer)
    rec_pos_cf = dfencoder_model_pos.decode_to_df(z_cer_pos)
    rec_neg_cf = dfencoder_model_neg.decode_to_df(z_cer_neg)
    
    im1_val = 0
    im2_val = 0
    
    for i in range(len(features)):
        im1_val = im1_score(features.loc[i], rec_pos_cf.loc[i], rec_neg_cf.loc[i])
        im2_val = im2_score(rec_pos_cf.loc[i], rec_cf.loc[i], features.loc[i])
        
    return im1_val/len(features), im2_val/len(features)

def multi_IM_calculate(features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg, method_name):
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
        im1_val = im1_score(features.loc[i], rec_pos_cf.loc[i], rec_neg_cf.loc[i])
        im2_val = im2_score(rec_pos_cf.loc[i], rec_cf.loc[i], features.loc[i])
        df_evaluation_im1.loc[loc_, 'value'] = im1_val
        df_evaluation_im1.loc[loc_, 'metric'] = 'IM1'
        df_evaluation_im1.loc[loc_, 'method'] = method_name   
        
        df_evaluation_im2.loc[loc_, 'value'] = im2_val
        df_evaluation_im2.loc[loc_, 'metric'] = 'IM2'
        df_evaluation_im2.loc[loc_, 'method'] = method_name   
        
    return df_evaluation_im1, df_evaluation_im2
    

def IM_evaluation(df_cf, df_evaluation, method_name, im1, im2):
    loc_ = len(df_evaluation)
    df_evaluation.loc[loc_, 'method'] = method_name
    df_evaluation.loc[loc_, 'metric'] = 'IM1'
    df_evaluation.loc[loc_, 'value'] = im1

    loc_ = len(df_evaluation)
    df_evaluation.loc[loc_, 'method'] = method_name
    df_evaluation.loc[loc_, 'metric'] = 'IM2'
    df_evaluation.loc[loc_, 'value'] = im2
    
    return df_evaluation

def causal_validity(df, dataset):
    c_validity = 0
    for i in range(len(df)):
        # if all(x  >= 0 for x in df.loc[i]) or all(x < 0 for x in df.loc[i]):
        if all(x <= 0.1 for x in df.loc[i]):
            c_validity += 1
    return c_validity / len(df)

def cau_val_evaluation(df_cf, df_original, df_evaluation, col, method_name, dataset):
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

def map_ordered_education(df):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='simple_bn')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    dataset_name = args.dataset
    seed = args.seed

    # dataset_name = 'siag'
    # dataset_name = 'simple_bn'
    # dataset_name = 'adult'
    
    """Outcome"""
    outcome = ''
    if dataset_name == 'adult':
        sub_col = ['age', 'new_education']
        outcome = 'income'
    elif dataset_name == 'simple_bn':
        sub_col = ['x1', 'x2']
        outcome = 'y'
    elif dataset_name == 'siag':
        sub_col = ['SproutN', 'BunchN']
        outcome = 'outcome'
    
    """Load result"""
    n_sample = 10
    DATA_PATH = cf.FINAL_RESULT_PATH.format(dataset_name + '/{}.csv')
    df_original = pd.read_csv(cf.ORG_PATH.format('full_negative_'+dataset_name+'.csv'))[:n_sample]
    df_certifai_org  = pd.read_csv(DATA_PATH.format('certifai'))[:n_sample]
    df_mad_org  = pd.read_csv(DATA_PATH.format('mad'))[:n_sample]
    df_mobj_org  = pd.read_csv(DATA_PATH.format('mobj'))[:n_sample]

    
    """Load auto encoder model"""    
    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format(dataset_name + '/dfencoder_full.pt'))
    dfencoder_model_pos = torch.load(cf.FINAL_MODEL_PATH.format(dataset_name + '/dfencoder_positive.pt'))
    dfencoder_model_neg = torch.load(cf.FINAL_MODEL_PATH.format(dataset_name + '/dfencoder_negative.pt'))
    
    emb_size = 32
    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format(dataset_name + '/dfencoder_full_{}.pt').format(emb_size))
    dfencoder_model_pos = torch.load(cf.FINAL_MODEL_PATH.format(dataset_name + '/dfencoder_positive_{}.pt').format(emb_size))
    dfencoder_model_neg = torch.load(cf.FINAL_MODEL_PATH.format(dataset_name + '/dfencoder_negative_{}.pt').format(emb_size))
            
    """Compute target-class validity"""    
    df_evaluation = pd.DataFrame()
    df_evaluation = target_valid(df_certifai_org, df_evaluation, 'certifai', outcome)
    df_evaluation = target_valid(df_mad_org, df_evaluation, 'mad', outcome)
    df_evaluation = target_valid(df_mobj_org, df_evaluation, 'mobj', outcome)

    """Filter"""
    df_certifai = df_certifai_org[df_certifai_org[outcome] >= 0.5].reset_index(drop=True)
    df_mad = df_mad_org[df_mad_org[outcome] >= 0.5].reset_index(drop=True)
    df_mobj = df_mobj_org[df_mobj_org[outcome] >= 0.5].reset_index(drop=True)
    
    """Get representation"""
    org_features = df_original.iloc[:,:-1]
    cer_features = df_certifai.iloc[:,:-1]
    mad_features = df_mad.iloc[:,:-1]
    mobj_features = df_mobj.iloc[:,:-1]

    # org_features = df_original.iloc[:,:-1]
    # cer_features = df_certifai_org.iloc[:,:-1]
    # mad_features = df_mad_org.iloc[:,:-1]
    # mobj_features = df_mobj_org.iloc[:,:-1]
    
    """Compute interpretability score"""
    # print("cer")
    im1_val, im2_val = IM_calculate(cer_features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg)
    df_evaluation = IM_evaluation(df_certifai, df_evaluation, 'certifai', im1_val, im2_val)
    # print(im1_val, im2_val)
    
    # print('mad')
    im1_val, im2_val = IM_calculate(mad_features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg)
    df_evaluation = IM_evaluation(df_mad, df_evaluation, 'mad', im1_val, im2_val)
    # print(im1_val, im2_val)

    # print('mobj')
    im1_val, im2_val = IM_calculate(mobj_features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg)
    df_evaluation = IM_evaluation(df_mobj, df_evaluation, 'mobj', im1_val, im2_val)
    # print(im1_val, im2_val)


    
    # sys.exit(1)
    method_name = 'cer'
    df_evaluation_im11, df_evaluation_im21 = multi_IM_calculate(cer_features, dfencoder_model, 
                                                              dfencoder_model_pos, dfencoder_model_neg, 
                                                              method_name)
    method_name = 'mad'
    df_evaluation_im12, df_evaluation_im22 = multi_IM_calculate(mad_features, dfencoder_model, 
                                                              dfencoder_model_pos, dfencoder_model_neg, 
                                                              method_name)
    method_name = 'mobj'
    df_evaluation_im13, df_evaluation_im23 = multi_IM_calculate(mobj_features, dfencoder_model, 
                                                              dfencoder_model_pos, dfencoder_model_neg, 
                                                              method_name)
    
    frames = [df_evaluation_im11, df_evaluation_im12, df_evaluation_im13]
    mul_con_prox = pd.concat(frames)  
    mul_con_prox = mul_con_prox.reset_index(drop = True)
    mul_con_prox.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/im1-" + dataset_name + ".csv"), index = False)

    # sys.exit(1)

    frames = [df_evaluation_im21, df_evaluation_im22, df_evaluation_im23]
    mul_con_prox = pd.concat(frames)  
    mul_con_prox = mul_con_prox.reset_index(drop = True)
    mul_con_prox.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/im2-" + dataset_name + ".csv"), index = False)

    # sys.exit(1)

    
    """Compute proximity"""
    if (dataset_name == 'siag') or (dataset_name == 'simple_bn'):  
        """Compute continous proximity"""
        df_evaluation = con_prox_eval(cer_features, org_features, df_evaluation, 'certifai')
        df_evaluation = con_prox_eval(mad_features, org_features, df_evaluation, 'mad')
        df_evaluation = con_prox_eval(mobj_features, org_features, df_evaluation, 'mobj') 
        
        mul_con_prox_cer = multiple_con_prox_eval(cer_features, org_features, 'certifai')
        df1 = pd.DataFrame()
        df1['value'] = mul_con_prox_cer
        df1['method'] = 'certifai'
        
        mul_con_prox_mad = multiple_con_prox_eval(mad_features, org_features, 'mad')
        df2 = pd.DataFrame()
        df2['value'] = mul_con_prox_mad
        df2['method'] = 'mad'
        
        mul_con_prox_mobj = multiple_con_prox_eval(mobj_features, org_features, 'mobj')
        df3 = pd.DataFrame()
        df3['value'] = mul_con_prox_mobj
        df3['method'] = 'mobj'
        
        frames = [df1, df2, df3]
        mul_con_prox = pd.concat(frames)  
        mul_con_prox = mul_con_prox.reset_index(drop = True)
        mul_con_prox.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/conprox-" + dataset_name + ".csv"), index = False)

        
    elif dataset_name == 'adult':
        con_indexes = [0,1]
        cat_indexes = [2,3,4,5,6,7]
        """Multi objective"""
        df_evaluation = con_prox_eval(df_mobj_org.iloc[:, con_indexes], 
                                      df_original.iloc[:,con_indexes],
                                                   df_evaluation,
                                                   'mobj')
        df_evaluation = cat_prox_eval(df_mobj_org.iloc[:, cat_indexes], 
                                      df_original.iloc[:, cat_indexes], df_evaluation, 'mobj')
        
        """MAD"""
        df_evaluation = con_prox_eval(df_mad_org.iloc[:, con_indexes], 
                                      df_original.iloc[:,con_indexes],
                                                   df_evaluation,
                                                   'mad')
        df_evaluation = cat_prox_eval(df_mad_org.iloc[:, cat_indexes], 
                                      df_original.iloc[:, cat_indexes], df_evaluation, 'mad')
        
        """Certifai"""
        df_evaluation = con_prox_eval(df_certifai_org.iloc[:, con_indexes], 
                                      df_original.iloc[:,con_indexes],
                                                   df_evaluation,
                                                   'certifai')
        df_evaluation = cat_prox_eval(df_certifai_org.iloc[:, cat_indexes], 
                                      df_original.iloc[:, cat_indexes], df_evaluation, 'certifai')

        mul_con_prox_cer = multiple_con_prox_eval(df_certifai.iloc[:, con_indexes], 
                                                  org_features, 'certifai')
        df1 = pd.DataFrame()
        df1['value'] = mul_con_prox_cer
        df1['method'] = 'certifai'
        
        mul_con_prox_mad = multiple_con_prox_eval(df_mad.iloc[:, con_indexes], 
                                                  org_features, 'mad')
        df2 = pd.DataFrame()
        df2['value'] = mul_con_prox_mad
        df2['method'] = 'mad'
        
        mul_con_prox_mobj = multiple_con_prox_eval(df_mobj.iloc[:, con_indexes],
                                                   org_features, 'mobj')
        df3 = pd.DataFrame()
        df3['value'] = mul_con_prox_mobj
        df3['method'] = 'mobj'
        
        frames = [df1, df2, df3]
        mul_con_prox = pd.concat(frames)  
        mul_con_prox = mul_con_prox.reset_index(drop = True)
        mul_con_prox.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/conprox-" + dataset_name + ".csv"), index = False)


        df1 = mul_cat_prox_eval(df_certifai.iloc[:, cat_indexes], df_original.iloc[:, cat_indexes], 'certifai')
        df2 = mul_cat_prox_eval(df_mad.iloc[:, cat_indexes], df_original.iloc[:, cat_indexes], 'mad')
        df3 = mul_cat_prox_eval(df_mobj.iloc[:, cat_indexes], df_original.iloc[:, cat_indexes], 'mobj')
        frames = [df1, df2, df3]
        mul_cat_prox = pd.concat(frames)  
        mul_cat_prox.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/catprox-" + dataset_name + ".csv"), index = False)
        
        # sys.exit(1)
        """Compute constraint valid"""
        df_original = map_ordered_education(df_original)
        df_certifai = map_ordered_education(df_certifai)
        df_mad = map_ordered_education(df_mad)
        df_mobj = map_ordered_education(df_mobj)
        
    df_evaluation = cau_val_evaluation(df_certifai, df_original, df_evaluation, sub_col, 
                                        'certifai', dataset_name)
    df_evaluation = cau_val_evaluation(df_mad, df_original, df_evaluation, sub_col,
                                        'mad', dataset_name)
    df_evaluation = cau_val_evaluation(df_mobj, df_original, df_evaluation, sub_col, 
                                        'mobj', dataset_name)
    # print(df_evaluation)
    df_evaluation.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/" + dataset_name + ".csv"), index = False)
    print("Output file to {}".format(cf.FINAL_RESULT_PATH.format("evaluation/" + dataset_name + ".csv")))
    
    
    """Different instance"""
    n_sample = 10
    emb_size = 512 

    df_evaluation = pd.DataFrame()
    if dataset_name == 'adult':
        length = 5
        start_index, end_index = 0, 5
    else:
        length = 4
        start_index, end_index = 0, 4
        
    df_evaluation = pd.DataFrame()
        
    for n_instance in range(5,50,5):
        DATA_PATH = cf.FINAL_RESULT_PATH.format(dataset_name + 
                                                '/different_instance/mobj-ninstance-{}-nsample-{}-size-{}.csv')
        df_mobj = pd.read_csv(DATA_PATH.format(n_instance, n_sample, emb_size))[:n_sample]
        mobj_features = df_mobj.iloc[:,:-1]
        df_evaluation = target_valid(df_mobj, df_evaluation, 'mobj', outcome)

        if (dataset_name == 'siag') or (dataset_name == 'simple_bn'):  
            """Compute continous proximity"""
            df_evaluation = con_prox_eval(mobj_features, org_features, df_evaluation, 'mobj')
        elif dataset_name == 'adult':
            con_indexes = [0,1]
            cat_indexes = [2,3,4,5,6]
            df_evaluation = cat_prox_eval(df_mobj.iloc[:, cat_indexes], 
                                          df_original.iloc[:, cat_indexes], df_evaluation, 'mobj')
            df_evaluation = con_prox_eval(df_mobj.iloc[:, con_indexes], 
                                          df_original.iloc[:,con_indexes],df_evaluation, 'mobj')
            """Compute constraint valid"""
            df_mobj = map_ordered_education(df_mobj)
        
        df_evaluation = cau_val_evaluation(df_mobj, df_original, df_evaluation, sub_col, 
                                        'mobj', dataset_name)
        im1_val, im2_val = IM_calculate(mobj_features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg)
        df_evaluation = IM_evaluation(df_mobj, df_evaluation, 'mobj', im1_val, im2_val)
        
        df_evaluation.loc[start_index:end_index, "n_instance"] = n_instance
        
        start_index = end_index + 1
        end_index = end_index + length + 1
        

    df_evaluation.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/different_instance-" + dataset_name + ".csv"), index = False)
   
            
    # sys.exit("Exit")
    
    
    """Different samples"""
    n_sample = 200
    n_instance = 10
    df_evaluation = pd.DataFrame()
    DATA_PATH = cf.FINAL_RESULT_PATH.format(dataset_name + 
                                            '/different_sample/mobj-ninstance-{}-nsample-{}-size-{}.csv')
    df_mobj = pd.read_csv(DATA_PATH.format(n_instance, n_sample, emb_size))[:n_sample]
    df_original = pd.read_csv(cf.ORG_PATH.format('full_negative_'+dataset_name+'.csv'))[:n_sample]

    for i in tqdm(range(1,n_sample+1)):
        df_evaluation = target_valid(df_mobj.loc[:i,:], df_evaluation, 'mobj', outcome)
    
        mobj_features = df_mobj.iloc[:i,:-1]
        if (dataset_name == 'siag') or (dataset_name == 'simple_bn'):  
            """Compute continous proximity"""
            df_evaluation = con_prox_eval(df_mobj.iloc[:i, :-1], 
                                          df_original.iloc[:i,:-1],df_evaluation, 'mobj')
        elif dataset_name == 'adult':
            con_indexes = [0,1]
            cat_indexes = [2,3,4,5,6]
            df_evaluation = cat_prox_eval(df_mobj.iloc[:i, cat_indexes], 
                                          df_original.iloc[:i, cat_indexes], df_evaluation, 'mobj')
            df_evaluation = con_prox_eval(df_mobj.iloc[:i, con_indexes], 
                                          df_original.iloc[:i,con_indexes],df_evaluation, 'mobj')

            df_original = map_ordered_education(df_original)
            df_certifai = map_ordered_education(df_certifai)
            df_mad = map_ordered_education(df_mad)
            df_mobj = map_ordered_education(df_mobj)
        
        df_evaluation = cau_val_evaluation(df_mobj.iloc[:i,:], df_original.iloc[:i,:], df_evaluation, sub_col, 
                                        'mobj', dataset_name)
        im1_val, im2_val = IM_calculate(mobj_features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg)
        df_evaluation = IM_evaluation(df_mobj.iloc[:i,:-1], df_evaluation, 'mobj', im1_val, im2_val)
    
    df_evaluation.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/different_sample-" + dataset_name + ".csv"), index = False)

    # sys.exit("Exit")

    """Different embedding size"""
    print("Evaluate different embedding size")
    n_sample = 10
    n_instance = 10
    df_original = pd.read_csv(cf.ORG_PATH.format('full_negative_'+dataset_name+'.csv'))[:n_sample]

    df_evaluation = pd.DataFrame()
    if dataset_name == 'adult':
        length = 5
        start_index, end_index = 0, 5
    else:
        length = 4
        start_index, end_index = 0, 4
    
    j = 1
    for emb_size in [32, 64, 128, 256, 512]:
        DATA_PATH = cf.FINAL_RESULT_PATH.format(dataset_name + '/different_size/mobj-ninstance-{}-nsample-{}-size-{}-seed-{}.csv')
        df_mobj = pd.read_csv(DATA_PATH.format(n_instance, n_sample,emb_size, seed))[:n_sample]
        df_evaluation = target_valid(df_mobj, df_evaluation, 'mobj', outcome)
        mobj_features = df_mobj.iloc[:,:-1]
        if (dataset_name == 'siag') or (dataset_name == 'simple_bn'):  
            """Compute continous proximity"""
            df_evaluation = con_prox_eval(mobj_features, org_features, df_evaluation, 'mobj')
        elif dataset_name == 'adult':
            con_indexes = [0,1]
            cat_indexes = [2,3,4,5,6]
            df_evaluation = cat_prox_eval(df_mobj.iloc[:, cat_indexes], 
                                          df_original.iloc[:, cat_indexes], df_evaluation, 'mobj')
            df_evaluation = con_prox_eval(df_mobj.iloc[:, con_indexes], 
                                          df_original.iloc[:,con_indexes],df_evaluation, 'mobj')
            
            df_original = map_ordered_education(df_original)
            df_certifai = map_ordered_education(df_certifai)
            df_mad = map_ordered_education(df_mad)
            df_mobj = map_ordered_education(df_mobj)
        df_evaluation = cau_val_evaluation(df_mobj, df_original, df_evaluation, sub_col, 
                                        'mobj', dataset_name)
        im1_val, im2_val = IM_calculate(mobj_features, dfencoder_model, dfencoder_model_pos, dfencoder_model_neg)
        df_evaluation = IM_evaluation(df_mobj, df_evaluation, 'mobj', im1_val, im2_val)
        df_evaluation.loc[start_index:end_index, "emb_size"] = emb_size
        
        j += 1
        start_index = end_index + 1
        end_index = end_index + length + 1
    df_evaluation.to_csv(cf.FINAL_RESULT_PATH.format("evaluation/different_size-{}-seed-{}-.csv".format(dataset_name, seed)), 
                         index = False)


    
    
 

    