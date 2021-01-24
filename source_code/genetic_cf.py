# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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





"""Pure distance"""
def pure_distance(x0, xcf):
    return mean_squared_error(x0, xcf)


"""Compute the continuous distance between two instances"""
def continous_dist(x0,xcf, d):
    loss = 0
    # for i in d.continuous_feature_indexes:
    loss += 5*(xcf[7]-x0[7])
    loss += 5*(xcf[0]-x0[0])
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
    dict_ = {'workclass': 1,
         'education': 2,
         'marital_status': 3,
         'occupation': 4}
    
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
    # print("FFFFFFFFFFFFF", xcf)
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
    # print("Term")
    # print(df_term)
    """Convert from label encoder back to the category name"""
    for c in d.categorical_feature_names:
        # print(c)
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
    # print(query)
    
    prepared_query = d.prepare_query_instance(query_instance=query,
                                                           encode = True)
    # print(query)

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
    mad_1 = np.mean(df['hours_per_week'].values - df['hours_per_week'].median())
    
    mad_2 = 4
    mad_3 = 6
    mad_4 = 1
    mad_5 = 2
    mad_6 = 3
    mad_7 = 2
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
    cat_indexes = [2,3,4,5,6,7]
    con_indexes = [0,1]
    loss = 0
    for i in cat_indexes:
        if x0[i] == xcf[i]:
            loss -= 1
    for j in con_indexes:
        loss += np.sqrt((xcf[j] - x0[j])**2)
    
    return loss 


if __name__ == "__main__":
    """Parsing argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', type=str, default='certifai')
    args = parser.parse_args()
    dist_name = args.distance
    # version = 0

    seed = 0
    torch.manual_seed(seed)   
    
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu" 
    device = torch.device(dev)  
    
    """Load data and dataloader and normalize data"""
    dataset = load_adult_income_dataset()
    params= {'dataframe':dataset.copy(), 
              'continuous_features':['age','hours_per_week'], 
              'outcome_name':'income'}
    d = DataLoader(params)
    df = d.data_df
    df = d.normalize_data(df)
    


    
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
    """Get category constraints"""    
    arr = []
    for i in [1,2,3,4]:
        arr.append(features.iloc[:,i].nunique())
        
    col = ['age','hours_per_week', 'workclass', 'education', 'marital_status', 'occupation', 'race',
       'gender']
    
    features = features[col]
    """Load prediction and auto encoder model"""    
    dfencoder_model = torch.load(cf.MODEL_FINAL_PATH.format('dfencoder.pt'))
    dfencoder_model_cat = torch.load(cf.MODEL_FINAL_PATH.format('categroy_subset_dfencoder.pt'))
        
    # pred_model = torch.load(cf.MODEL_FINAL_PATH.format('dfencoder_pred_model.pt'))
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
    
    # sys.exit("Age less than 18")     
    list_ = neg_index.detach().cpu().numpy().reshape(-1)



        
    """Parameter setup"""
    threshold = -0.5
    num_generations, num_parents_mating, sol_per_pop =  cf.num_generations, cf.num_parents_mating, cf.sol_per_pop
    init_range_low, init_range_high = cf.init_range_low, cf.init_range_high
    parent_selection_type, keep_parents = cf.parent_selection_type, cf.keep_parents
    crossover_type, mutation_type, mutation_percent_genes = cf.crossover_type, cf.mutation_type,cf.mutation_percent_genes
    mutation_probability = cf.mutation_probability
    
    last_fitness = 0
    
    list_ = neg_index.detach().cpu().numpy().reshape(-1)    
    
    df_original = pd.DataFrame(columns = list(features.columns))
    df_counterfactual = pd.DataFrame(columns = list(features.columns))
    df_original['income'] = -1
    df_counterfactual['income'] = -1
    
    file_name = "genetic_version_{}.log".format(dist_name)
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
    
    n_sample = 10
    for i in range(n_sample):
        index_ = list_[i]
    # for index_ in list_:
        """Global variable"""    
        x_valid = np.array([0]*8)
        best_fitness = 0
        best_prediction_score = 0
        pureDist = 10000
        
        df_store = pd.DataFrame(columns = list(features.columns))

        logger.debug("Sample index {}".format(i))
        
        """Get original instance
            query = {'age': 39,
                 'workclass': 'Government',
                 'education': 'Bachelors',
                 'marital_status': 'Single',
                 'occupation': 'White-Collar',
                 'race': 'White',
                 'gender': 'Male',
                 'hours_per_week': 40}
        
            prepared_query = [0.30136986, 0.39795918, 1.        , 0.        , 0.        ,
                   0.        , 0.        , 1.        , 0.        , 0.        ,
                   0.        , 0.        , 0.        , 0.        , 0.        ,
                   0.        , 0.        , 1.        , 0.        , 0.        ,
                   0.        , 0.        , 0.        , 0.        , 1.        ,
                   0.        , 1.        , 0.        , 1.        ]   
            
            x0 = [0.30136986, 0.        , 1.        , 3.        , 5.        ,
       1.        , 1.        , 0.39795918]
            xcf = [0.30136986, 0.        , 1.        , 3.        , 5.        ,
       1.        , 1.        , 0.39795918]
        
        """
        # query = features.iloc[index_,:].to_dict()
        # one_hot_query = convert_dictionary_one_hot(query, d)
        # label_encoded_query = label_encoder_instance(query, d)
        query = features.iloc[index_,:].to_dict()
        x0 = list(query.values())
        
        # x0 = convert_query_to_label(label_encoded_query)
        # print("Original instance", x0)
        # sys.exit("Age less than 18")     
        # sys.exit("Age less than 18")     
        
        """Logging info """
        logger.debug("Original instance {}".format(x0))
        logger.debug("----------------------------------------")

        one_hot_query = convert_label_to_ohe(x0, d)
        one_hot_query = np.array(one_hot_query)
        tensor0 = torch.from_numpy(one_hot_query).float().to(device)
        # df_store.loc[0] = x0
        # z = dfencoder_model.get_representation(df_store)
        # z0 = z[0]

        
        # sys.exit("Age less than 18")     
        
        """Find prototype"""
        # k_instance=  5
        # pos_proto, neg_proto = find_proto(z0.reshape(1,-1), pos_z, neg_z, k_instance)
    
        """Number of genes"""
        num_genes = len(x0)
        
        """Fitness function"""
        def fitness_func(xcf, solution_idx):
            df_store.loc[0] = x0
            df_store.loc[1] = xcf   
            
            # df_store.to_csv(cf.RESULT_PATH.format("genetic_result.csv"),index = False)
            
            xcf[0] = max(xcf[0],0.1)
            xcf[0] = min(xcf[0],0.8)
            xcf[1] = max(xcf[1],0.1)
            xcf[1] = min(xcf[1],0.8)
                        
            """Get the representation"""
            z = dfencoder_model.get_representation(df_store)
            # z0 = z[0]
            # zcf = z[1] 

            z_cat = dfencoder_model_cat.get_representation(df_store)
            # z0_cat = z_cat[0]
            # zcf_cat = z_cat[1] 

            """Compute the pure distance"""
            # pure_dist = pure_distance(x0, xcf, d)
            
            """Compute continuous distance"""
            # con_dist = continous_dist(x0, xcf, d)
                        
            """Compute latent distance"""
            # l_distance = latent_distance(z0, zcf)
            # l_distance = latent_distance(z0_cat, zcf_cat)
            
            """Compute prediction loss"""
            ycf = torch.Tensor([1.0]).to(device)
            yloss = compute_yloss(xcf, ycf, pred_model, d)
            
            # print("Yloss ", yloss)
            """Compute prototype loss"""
            # ploss = proto_loss(zcf, pos_proto)
            
            """Compute causal loss"""
            # cau_loss = causal_loss(x0, xcf)
                        
            """Compute constraint loss"""            
            # tensor0 = torch.from_numpy(np.array(x0)).float().to(device)
            # tensor_cf = torch.from_numpy(np.array(xcf)).float().to(device)
            # closs = constraints_loss(tensor0, tensor_cf)
            
            """Compute categorical similarity"""
            # similarity = cat_representation_dist(dfencoder_model, x0, xcf)
            
            """Certifai distance"""
            
            
            """MAD distance"""
            
            
            """Compute final loss"""
            if dist_name == 'certifai':
                certifai = certifai_distance(x0, xcf)
                final_loss = certifai + 2*yloss
            else:
                maddist = distance_mad(x0, xcf, df)
                final_loss = 3*maddist + 2*yloss
            # if version == 0:
                # final_loss = 4*yloss - similarity + con_dist + ploss
                # final_loss = 7*yloss - 2*similarity + 3*con_dist + 0.5*ploss + 0.01*closs
                # final_loss = 7*yloss - 0.02*similarity + 0.03*con_dist
                # final_loss = 7*yloss - 0.02*similarity

                
                # final_loss = 7*yloss - 2*similarity + 4*con_dist + 0.5*ploss + 0.01*closs
                # print("Similarity ", similarity)
                # print("yloss ", yloss)
                # print("con_dist ", con_dist)
                # print("Final ", final_loss)
                # print("---------------")

            # elif version == 1:
            #     final_loss = 0.4*yloss - similarity + con_dist
            # elif version == 2:
            #     final_loss = 4*yloss - similarity + con_dist + closs
            # elif version == 3:
            #     final_loss = 4*yloss - similarity + con_dist + ploss
            # elif version == 4:
            #     final_loss = 4*yloss - similarity + con_dist + ploss + closs
            # elif version == 5:
            #     final_loss = 4*yloss - similarity + con_dist + ploss + cau_loss
            # elif version == 6:
            #     final_loss = 4*yloss - similarity + con_dist + ploss
            # elif version == 7:
            #     final_loss = 4*yloss - similarity + 7*con_dist
            # elif version == 8:
            #     final_loss = 4*yloss - similarity + 7*con_dist + ploss
            return -final_loss
        
        """Callback function"""
        def callback_generation(ga_instance):
            global x_valid
            global pureDist
            global last_fitness

            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            
            """Logging"""
            fitness = ga_instance.best_solution()[1]
            change = ga_instance.best_solution()[1] - last_fitness
            logger.debug("Generation = {generation}".format(generation=ga_instance.generations_completed))
            logger.debug("Fitness {:.4f} and Change {:.4f}".format(fitness,change))


            last_fitness = ga_instance.best_solution()[1]
            logger.debug("Solution {}".format(solution))
            
            logger.debug("Validation solution {}".format(x_valid))
            
            """Get the prediction for counterfactual sample"""
            df_store.loc[1] = x_valid 
            df_store_term = convert_to_understandable(d, df_store)    
            
            p_dist = pure_distance(x0, solution)
            
            one_hot_solution = convert_label_to_ohe(solution, d)
            tensor_cf = torch.from_numpy(one_hot_solution).float().to(device)                
            prediction = torch.ge(pred_model(tensor_cf), 0.5).int()
            
            """Check if the conunterfactual belong to opposite class"""
            main_condition = bool(prediction.detach().cpu().numpy()[0])        
            logger.debug("Prediction {}".format(pred_model(tensor_cf)))
            logger.debug("Current distance {}".format(pureDist))

            logger.debug('Compare') 
            print('{}'.format(df_store_term.to_string()))            
            if (main_condition) & (pureDist > p_dist):
                term_solution = copy.deepcopy(solution)
                x_valid = term_solution
                term_x_valid = copy.deepcopy(x_valid)
                one_hot_solution = convert_label_to_ohe(term_x_valid, d)
                tensor_cf = torch.from_numpy(one_hot_solution).float().to(device)
                pureDist = p_dist
                del p_dist
            logger.debug("*************************************")

        """Creating an instance of the GA class inside the ga module."""
        ga_instance = pygad.GA(
                               initial_population = [x0]*sol_per_pop,
                               num_generations=num_generations,
                               num_parents_mating=num_parents_mating, 
                               fitness_func=fitness_func,
                               sol_per_pop=sol_per_pop, 
                               num_genes=num_genes,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes,
                               callback_generation=callback_generation,
                                mutation_probability = mutation_probability,
                                gene_space=[None, 
                                            None,
                                            range(arr[0]), 
                                            range(arr[1]),
                                            range(arr[2]),
                                            range(arr[3]),
                                            int(x0[6]),
                                            int(x0[7])
                                            ],
                                random_mutation_min_val=0,
                                random_mutation_max_val=1.0
                               )
        
        ga_instance.run()
        
        """Returning the details of the best solution."""
        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        """Logging"""
        logger.debug("Parameters of the best solution : {solution}".format(solution=solution))
        logger.debug("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        logger.debug("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
        
        
        if type(x_valid) != int:
            logger.debug("Found the valid solution")
            logger.debug(x_valid)
            solution = x_valid
        else:
            logger.debug("No solution")

        
        df_store.loc[1] = solution
        logger.debug("Final result")
        logger.debug(df_store)

        df_counterfactual.at[index_, :-1] = solution
        df_original.at[index_, :-1] = x0

        """Check the final prediction"""
        pred0 = pred_model(tensor0)
        solution = convert_label_to_ohe(solution, d)
        tensor_solution = torch.from_numpy(solution).float().to(device)
        pred1 = pred_model(tensor_solution)
    
        """Convert from label encoder back to the category name"""
        """Denomalize dataframe"""
        df_counterfactual.at[index_, 'income'] = pred1.detach().cpu().numpy()
        df_original.at[index_, 'income'] = pred0.detach().cpu().numpy()
        
        logger.debug("Final result")
        logger.debug(df_counterfactual)
        logger.debug("Done!!!!!!!!!")
        logger.debug("***********************************************************************")

        del df_store
        del ga_instance
        # break
    
    torch.cuda.empty_cache()

    df_original = df_original.round(3)
    df_counterfactual = df_counterfactual.round(3)

    logger.debug("Ouput to file to {}".format(cf.FINAL_RESULT_PATH.format("/adult/certifai.csv")))
    logger.debug("***********************************************************************")
    logger.debug("***********************************************************************")
    logger.debug("\n")

    print("Output file to ", cf.FINAL_RESULT_PATH.format("adult/"+dist_name+".csv"))
    # version_name = 'version_{}'.format(version)
    df_original.to_csv(cf.FINAL_RESULT_PATH.format("adult/"+dist_name+"original.csv"),
                       index = False)
    df_counterfactual.to_csv(cf.FINAL_RESULT_PATH.format("adult/"+dist_name+".csv"),
                             index = False)
