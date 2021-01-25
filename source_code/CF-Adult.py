"""
All the views for our todos application
Currently we support the following 3 views:

1. **Home** - The main view for Todos
2. **Delete** - called to delete a todo
3. **Add** - called to add a new todo

@author: trduong

"""

import sys
MAIN_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm"
sys.path.insert(1, MAIN_PATH)

import pygad
import numpy as np
import pandas as pd
import copy
import logging 
import argparse
import pickle
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import torch 
import torch.nn as nn

from utils.helpers import load_adult_income_dataset
from utils.dataloader import DataLoader
from source_code.prototype import find_proto, get_pos_neg_latent
from dfencoder.autoencoder import AutoEncoder
import source_code.configuration_path as cf
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report

import autograd.numpy as anp

from pymoo.model.problem import Problem
from pymoo.util.normalization import normalize
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
import matplotlib.pyplot as plt
from pymoo.util import plotting

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
device = torch.device(dev)  
    

"""Pure distance"""
def pure_distance(x0, xcf):
    cat_indexes = [1,2,3,4]
    con_indexes = [0,7] 
    
    cat_sim = 0
    con_sim = 0
    for i in cat_indexes:
        if x0[i] ==  xcf[i]:
            cat_sim += 1
            
    #cat_sim /= 4
    
    for j in con_indexes:
        con_sim -= np.sqrt((xcf[j]-x0[j])**2)
    
    #con_sim /= 2
        
    return cat_sim + con_sim


"""Compute the continuous distance between two instances"""
def continous_dist(x0,xcf, d):
    loss = 0
    loss += np.sqrt((xcf[7]-x0[7])**2)
    loss += np.sqrt((xcf[0]-x0[0])**2)
    return loss 

"""Compute the distance in the latent space"""
def latent_distance(z0, zcf):
    criterion =  nn.MSELoss()
    dist = criterion(z0,zcf).detach().cpu().numpy()
    return dist 


"""Compute distance between two categories in dataset"""
def two_cate_dist(model, feature, cat1, cat2):
    feature = model.categorical_fts[feature]
    cats = feature['cats']
    cat1_index = cats.index(cat1)
    cat2_index = cats.index(cat2)
    
    emb = feature['embedding']    
    emb_cat1 = emb.weight.data.cpu().numpy()[cat1_index, :]    
    emb_cat2 = emb.weight.data.cpu().numpy()[cat2_index, :]
    similarity = cosine(emb_cat1, emb_cat2)
        
    return similarity 


"""Compute the categorical distance in latent space"""
def cat_representation_dist(dfencoder_model, x0, xcf):
    dict_ = {'workclass': 2,
         'education': 3,
         'marital_status': 4,
         'occupation': 5}
    
    similarity = 0
    for k, v in dict_.items():
        similarity += two_cate_dist(dfencoder_model, k, x0[v], xcf[v])
    
    return similarity
    


"""Compute the prototype loss function"""
def proto_loss(zcf, proto):   
    criterion = nn.MSELoss()
    proto_loss = criterion(proto.reshape(-1), zcf)
    proto_loss = proto_loss.detach().cpu().numpy()
    return proto_loss

"""Compute the constraint loss: Age have to be larger than original instance"""
def constraints_loss(x0, xcf):
    criterion = nn.MarginRankingLoss(margin = 0)
    loss1 = criterion(xcf[0].reshape(-1,1), x0[0].reshape(-1,1), torch.Tensor([1]).to(device))
    # criterion = nn.MarginRankingLoss(margin = 0)
    # loss2 = criterion(torch.Tensor([50]).to(device), x0[0].reshape(-1,1), torch.Tensor([1]).to(device))
    # criterion = nn.MarginRankingLoss(margin = 1)
    # loss3 = criterion(xcf[7].reshape(-1,1), x0[7].reshape(-1,1), torch.Tensor([1]).to(device))


    loss = loss1
    loss = loss.detach().cpu().numpy()
    return loss 

"""Compute the prediction loss function"""
def compute_yloss(xcf, ycf, prediction_model, d):
    term_xcf = convert_label_to_ohe(xcf, d)
    tensor_cf = torch.from_numpy(term_xcf).float().to(device)
    ycf_pred = prediction_model(tensor_cf)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(ycf, ycf_pred)    
    loss = loss.detach().cpu().numpy()
    return loss

"""Compute the distance with structural causal model"""
def causal_loss(x0, xcf):
    x0 = np.array(x0)
    xcf = np.array(xcf)
    subset_dfencoder = torch.load(cf.MODEL_FINAL_PATH.format('subset_dfencoder.pt'))
    
    dict_ = {'age': xcf[0],
             'workclass': int(xcf[1]),
             'education': int(xcf[2]),
             'marital_status': int(xcf[3]),
             'occupation': int(xcf[4]),
             'race': int(xcf[5]),
             'gender': int(xcf[6]),
             'hours_per_week': xcf[7]}
    
    c = 'occupation'
    le = LabelEncoder()
    le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    mapping1 = {v: k for k, v in mapping.items()}
    mapping2 = {k: v for k, v in mapping.items()}
    
        
    dictionary_map = {
    'Blue-Collar': 1,
    'Other/Unknown': 1,
    'Professional': 2,
    'White-Collar': 2,
    'Sales': 1,
    'Service': 1
    }
    
    map_back = {
        2: 'White-Collar',
        1: 'Sales'
    }
    
    
    
    df_term = pd.DataFrame({k: [v] for k, v in dict_.items()})
    df_term = df_term[['age', 'race', 'gender', 'education', 'workclass', 'marital_status']]

    z = subset_dfencoder.get_representation(df_term)
    z = z.detach().cpu().numpy()

    # clf = pickle.loads(cf.MODEL_FINAL_PATH.format('occupation_binary_classifier.pkl'))
    clf = pickle.load(open(cf.MODEL_FINAL_PATH.format('occupation_binary_classifier.pkl'), 'rb'))

    prediction = clf.predict(z)
    prediction = mapping2[map_back[prediction[0]]]
    
    tensor0 = torch.from_numpy(np.array([x0[4]])).float().to(device)
    tensor_cf = torch.from_numpy(np.array([prediction])).float().to(device)
            
    criterion = nn.MarginRankingLoss(margin = 0.01)
    loss = criterion(tensor0, tensor_cf, torch.Tensor([0]).to(device))
    
    return loss.detach().cpu().numpy()
   

def convert_to_understandable(d, df_store):
    df_term = df_store.copy()
    """Convert from label encoder back to the category name"""
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        df_term[c] = df_term[c].astype(int)
        df_term[c] = list(le.inverse_transform(df_term[c].values))
        
    """Denomalize dataframe"""
    df_term = d.de_normalize_data(df_term)
    return df_term 


def convert_to_understandable_nonnormoalize(d, df_store):
    df_term = df_store.copy()
    """Convert from label encoder back to the category name"""
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        df_term[c] = df_term[c].astype(int)
        df_term[c] = list(le.inverse_transform(df_term[c].values))
        
    """Denomalize dataframe"""
    df_term = d.de_normalize_data(df_term)
    return df_term 



def label_encoder_instance(query, d):
    """Label encoder
    query = {'age': 39,
            'workclass': 'Government',
            'education': 'Bachelors',
            'marital_status': 'Single',
            'occupation': 'White-Collar',
            'race': 'White',
            'gender': 'Male',
            'hours_per_week': 40}
    
    output =  {'age': 39,
            'workclass': 1,
            'education': 2,
            'marital_status':  4,
            'occupation': 8,
            'race': 2,
            'gender': 2,
            'hours_per_week': 40}

    """
    c_query = copy.deepcopy(query)
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        c_query[c] = le.transform([c_query[c]])[0]
    return c_query 
 
def convert_query_to_label(query):
    """
        input =  {'age': 39,
            'workclass': 1,
            'education': 2,
            'marital_status':  4,
            'occupation': 8,
            'race': 2,
            'gender': 2,
            'hours_per_week': 40}
        
        output = [39,40,1,2,4,8,2,2]
    
    """
    c_query = copy.deepcopy(query)
    return [value for key, value in c_query.items()]

def convert_dictionary_one_hot(query, d):
    col = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'race',
       'gender','hours_per_week']
    df_term = pd.DataFrame({k: [v] for k, v in query.items()})
    df_term = df_term[col]
    # df_term = d.de_normalize_data(df_term)
    query = df_term.loc[0].to_dict()    
    prepared_query = d.prepare_query_instance(query_instance=query,
                                                           encode = True)
    return prepared_query.loc[0].values

def convert_label_to_ohe(x0, d):
    """Input: [0.3013698630136986, 0, 1, 3, 5, 1, 1, 0.3979591836734694]
       Ouput: 
    """
    
    dict_ = {'age': x0[0],
             'hours_per_week': x0[1],
             'workclass': int(x0[2]),
             'education': int(x0[3]),
             'marital_status': int(x0[4]),
             'occupation': int(x0[5]),
             'race': int(x0[6]),
             'gender': int(x0[7])}
    
    df_term = pd.DataFrame({k: [v] for k, v in dict_.items()})
    df_term = convert_to_understandable(d, df_term)
    dict_ = df_term.loc[0].to_dict()
    ohe_x0 = convert_dictionary_one_hot(dict_, d)
    return ohe_x0
    
"""Compute the median absolute deviation in training data"""
def MAD(df):    
    mad_0 = np.mean(df['age'].values - df['age'].median())
    mad_7 = np.mean(df['hours_per_week'].values - df['hours_per_week'].median())
    
    mad_1 = 4
    mad_2 = 6
    mad_3 = 1
    mad_4 = 2
    mad_5 = 3
    mad_6 = 2
    return [mad_0, mad_1, mad_2, mad_3, mad_4, mad_5, mad_6, mad_7]

"""Compute loss for MAD"""
def distance_mad(x0, xcf, df):
    """MAD features importance"""
    mean_absolute_deviation = np.array(MAD(df))   
    
    """distance"""
    distance = (x0-xcf)/mean_absolute_deviation
    
    return np.mean(distance) 

"""Compute loss for CERTIFAI"""
def certifai_distance(x0, xcf):
    cat_indexes = [1,2,3,4,5,6]
    con_indexes = [0,7]
    loss = 0
    for i in cat_indexes:
        if x0[i] == xcf[i]:
            loss -= 1
    for j in con_indexes:
        loss += (xcf[j] - x0[j])
    
    return loss 


"""
Number of variable: 30
Number of objective function: 2
"""
class CF2(Problem):
    def __init__(self, x0, 
                 d, 
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
                         xl=np.array([0,0,0,0,0,0,x6,x7]), 
                         xu=np.array([0.95,0.85,3,7,4,5,x6,x7]),
                         type_var=anp.double,
                         elementwise_evaluation=True,
                         **kwargs)
        self.x0 = x0
        self.pred_model = pred_model
        self.proto = pos_proto
        self.dfencoder_model = dfencoder_model

    def _evaluate(self, xcf, out, *args, **kwargs):
        df_store = pd.DataFrame(columns = list(features.columns))
        
        df_store.loc[0] = x0
        df_store.loc[1] = xcf   
        """Get the representation"""
        z = dfencoder_model.get_representation(df_store)
        zcf = z[1] 
            
        """Compute prediction loss"""
        ycf = torch.Tensor([1.0]).to(device)
        yloss = compute_yloss(xcf, ycf, self.pred_model, d)
        con_dist = continous_dist(self.x0, xcf, d)
        cat_dist = cat_representation_dist(self.dfencoder_model, self.x0, xcf)
        
        """Compute prototype loss"""
        ploss = proto_loss(zcf, self.proto)        
        
        """Constraints"""
        g1 = x0[0] - xcf[0]
        
        out["F"] = anp.column_stack([yloss, con_dist, cat_dist, ploss])
        # out["F"] = anp.column_stack([yloss, con_dist, cat_dist])
        out["G"] = np.column_stack([g1])
        
        
if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='different_instance')
    parser.add_argument('--n_instance', type=int, default=10)
    parser.add_argument('--n_sample', type=int, default=10)
    parser.add_argument('--emb_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()
    mode = args.mode
    n_instance = args.n_instance
    n_sample = args.n_sample
    emb_size = args.emb_size
    seed = args.seed
    
    torch.manual_seed(seed)   
    
    
    """Load data and dataloader and normalize data"""
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 
              'continuous_features':['age','hours_per_week'], 
              'outcome_name':'income'}
    d = DataLoader(params)
    df = d.data_df
    df = d.normalize_data(df)
    
    sys.exit(1)
    """Preprocess data"""
    encoded_data = d.one_hot_encoded_data
    encoded_data = d.normalize_data(encoded_data)
    encoded_features = encoded_data.drop(columns = ['income'])
    encoded_features_tensor = torch.from_numpy(encoded_features.values).float().to(device)
    features = df.drop(columns = ['income'])

    for v in d.categorical_feature_names:
        features[v] = pd.Categorical(features[v].values)
    
    for c in d.categorical_feature_names:
        le = LabelEncoder()
        le.classes_ = np.load(cf.MODEL_FINAL_PATH.format(c + '.npy'), allow_pickle=True)
        features[c] = le.fit_transform(features[c].values)

    col = ['age','hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'race',
       'gender']
    
    features = features[col]
    """Load prediction and auto encoder model"""    
    name = 'adult'
    version = 'full'
    # dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format('/' + name + '/dfencoder_{}.pt'.format(version)))
    print("Load autoencoder model from {}".format(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size))))
    dfencoder_model = torch.load(cf.FINAL_MODEL_PATH.format(name + '/dfencoder_{}_{}.pt'.format(version,emb_size)))

    pred_model = torch.load('/data/trduong/DiCE/dice_ml/utils/sample_trained_models/adult.pth')
    pred_model.to(device)
    
    """Store the result"""
    df_store = pd.DataFrame(columns = list(features.columns))

    """Get the representation for the dataset"""
    z_representation = dfencoder_model.get_representation(features)

    prediction_score = pred_model(encoded_features_tensor)
    prediction = torch.ge(prediction_score, 0.5).int()
    prediction = prediction.reshape(-1)
    
    neg_index = (prediction == 0).nonzero()
    
    pos_z, neg_z = get_pos_neg_latent(prediction, z_representation)
    
    """Get category constraints"""    
    arr = []
    for i in d.categorical_feature_indexes:
        arr.append(features.iloc[:,i].nunique())
        

    
    list_ = neg_index.detach().cpu().numpy().reshape(-1)

    ORG_PATH = "/data/trduong/counterfactual-explanation-research.git/my-algorithm/final_result/result/original/{}"
    neg_features = features.iloc[list_,:]
    neg_features['income'] = 0
    neg_features.to_csv(ORG_PATH.format('full_negative_adult.csv'), index = False)
    # sys.exit(1)
    

    
    """Set up logging"""
    file_name = "multi_objective_{}.log".format(version)
    PATH = cf.LOGGING_PATH.format(file_name)
    logger = logging.getLogger('genetic')
    file_handler = logging.FileHandler(filename=PATH)
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logger.setLevel(logging.DEBUG)
    
    loc_= 0 
    # n_sample = n_sample

    df_original = pd.DataFrame(columns = list(features.columns))
    df_counterfactual = pd.DataFrame(columns = list(features.columns))
    df_original['income'] = -1
    df_counterfactual['income'] = -1
    
    count_ = 0
    for index_ in list_[:n_sample]:
        count_ +=  1
        # df_original = pd.DataFrame(columns = list(features.columns))
        # df_counterfactual = pd.DataFrame(columns = list(features.columns))
    
        """Global variable"""    
        x_valid = 0
        best_fitness = 0
        best_prediction_score = 0
        pureDist = 10000
        
        df_store = pd.DataFrame(columns = list(features.columns))

        logger.debug("Sample index {}".format(count_))
        
        mask = ["real","real","int", "int", "int", "int", "int", "int"]
        sampling = MixedVariableSampling(mask, {
            "real": get_sampling("real_random"),
            "int": get_sampling("int_random"),
        })
        
        crossover = MixedVariableCrossover(mask, {
            "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
            "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
        })
        
        mutation = MixedVariableMutation(mask, {
            "real": get_mutation("real_pm", eta=3.0),
            "int": get_mutation("int_pm", eta=3.0)
        })

        """Get original instance"""
        query = features.iloc[index_,:].to_dict()
        x0 = list(query.values())
        df_store.loc[0] = x0
        z = dfencoder_model.get_representation(df_store)
        z0 = z[0]

        
        # sys.exit("Age less than 18")     
        
        """Find prototype"""
        k_instance = n_instance
        pos_proto, neg_proto = find_proto(z0.reshape(1,-1), pos_z, neg_z, k_instance)
        
        problem = CF2(x0, 
                      d, 
                      pred_model, 
                      dfencoder_model, 
                      pos_proto,
                      x0[6],
                      x0[7]
                      )
        
        algorithm = NSGA2(
                pop_size=100,
                n_offsprings=50,
                sampling=sampling,
                crossover=crossover,
                mutation=mutation,
                eliminate_duplicates=True)
        
        res = minimize(problem,
                        algorithm,
                        ('n_gen', 30),
                        save_history=True,
                        seed = seed,
                        verbose=True)
    

        x_valid = 0
        y_valid = -1
        current_dist = -9999
        index_ = -1
        df_store = pd.DataFrame(columns = list(features.columns))
    
    
        for i in range(len(res.X)):
            xcf = res.X[i]
            term_xcf = convert_label_to_ohe(xcf, d)
            tensor_cf = torch.from_numpy(term_xcf).float().to(device)
            ycf_pred = pred_model(tensor_cf)
            if ycf_pred >= 0.5:
                y_valid = ycf_pred.detach().cpu().numpy()
                if pure_distance(x0, xcf) >= current_dist:
                    current_dist = pure_distance(x0, xcf)
                    x_valid = xcf
                    index_ = i
        df_counterfactual.at[loc_, :-1] = x_valid
        if type(y_valid) != int:
            df_counterfactual.at[loc_, 'income'] = y_valid[0]
        else:
            df_counterfactual.at[loc_, 'income'] = y_valid
            
        df_original.at[loc_, :-1] = x0
        df_original.at[loc_, 'income'] = 0
        loc_ += 1
        print(df_original)
        print(df_counterfactual)
        df_counterfactual.to_csv( cf.FINAL_RESULT_PATH.format("adult/"+mode +"/mobj-ninstance-{}-nsample-{}-size-{}.csv".format(n_instance, n_sample, emb_size)),
                              index = False)
        # break

    print("Output file to ", cf.FINAL_RESULT_PATH.format("adult/"+ mode + "/mobj-ninstance-{}-nsample-{}-size-{}.csv".format(n_instance, n_sample, emb_size)))
    # print("Output file to ", cf.FINAL_RESULT_PATH.format("adult/original.csv"))    
    
    # df_original.to_csv(cf.FINAL_RESULT_PATH.format("adult/original.csv"),
    #                          index = False)    
    df_counterfactual.to_csv( cf.FINAL_RESULT_PATH.format("adult/"+mode +"/mobj-ninstance-{}-nsample-{}-size-{}-seed-{}.csv".format(n_instance, n_sample, emb_size, seed)),
                              index = False)

