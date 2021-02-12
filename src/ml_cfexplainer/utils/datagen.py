#Important Classes
from dataloader import DataLoader
from helpers import *
import logging

#Normie stuff
import sys
import pandas as pd
import numpy as np
import json

base_dir='/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/data/'
dataset_name = sys.argv[1]

# Logging
LOG_PATH = "/home/trduong/Data/counterfactual-explanation-research.git/my-algorithm/logs"
logging.basicConfig(filename=LOG_PATH + "/datagen.log", filemode='w', format='%(name)s - %(levelname)s - %(message)s')



#Dataset
if dataset_name=='bn1':
    dataset= pd.read_csv(base_dir+dataset_name+'.csv')
    dataset.drop(['Unnamed: 0'], axis=1, inplace=True)	

    params= {'dataframe':dataset.copy(), 'continuous_features':['x1','x2','x3'], 'outcome_name':'y'}
    d = DataLoader(params)
    train_data_vae= d.data_df.copy()
    #train_data_vae.drop('y', axis=1, inplace=True)
    columns= train_data_vae.columns   

elif dataset_name=='adult':
    dataset = load_adult_income_dataset()

    params= {'dataframe':dataset.copy(), 'continuous_features':['age','hours_per_week'], 'outcome_name':'income'}
    d = DataLoader(params)
    # d = dice_ml.Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    train_data_vae= d.data_df.copy()
    #train_data_vae= train_data_vae[ train_data_vae['income']==0 ]
    #train_data_vae.drop('income', axis=1, inplace=True)
    
    train_data_vae_l0 = train_data_vae[ train_data_vae['income']==0]
    train_data_vae_l0 = train_data_vae_l0[train_data_vae_l0['age'] > 35]
    train_data_vae_l1 = train_data_vae[train_data_vae['income']==1]
    train_data_vae_l1 = train_data_vae_l1[train_data_vae_l1['age'] < 45]
    train_data_vae = pd.concat([train_data_vae_l0, train_data_vae_l1], axis=0 )
    
    columns= train_data_vae.columns
    
elif dataset_name=='sangiovese':
    dataset = pd.read_csv(  base_dir + dataset_name + '.csv', index_col=None )
    dataset= dataset.drop(columns= ['Unnamed: 0'])
    outcome=[]
    for i in range(dataset.shape[0]):
        if dataset['GrapeW'][i] > 0: 
            outcome.append( 1 )
        else:
            outcome.append( 0 )
    dataset['outcome'] = pd.Series(outcome)
    dataset.drop(columns=['GrapeW'], axis=1, inplace=True)

    # Continuous Features
    l=list(dataset.columns)
    l.remove('outcome')

    params= {'dataframe':dataset.copy(), 'continuous_features':l, 'outcome_name':'outcome'}
    d = DataLoader(params)
    train_data_vae= d.data_df.copy()
    #train_data_vae.drop('outcome', axis=1, inplace=True)
    columns= train_data_vae.columns   

logging.warning("Dataset {}".format(dataset_name))
logging.warning("Dataset {}".format(train_data_vae.head(20)))


#MAD
mad_feature_weights = d.get_mads_from_training_data(normalized=False)
logging.warning("Weight {}".format(mad_feature_weights))

#One Hot Encoding for categorical features
encoded_data = d.one_hot_encode_data(train_data_vae)
dataset = encoded_data.to_numpy()

#Normlaise_Weights
data_size = len(d.encoded_feature_names)
encoded_categorical_feature_indexes = d.get_data_params()[2]     
encoded_continuous_feature_indexes=[]
for i in range(data_size):
    valid=1
    for v in encoded_categorical_feature_indexes:
        if i in v:
            valid=0
    if valid:
        encoded_continuous_feature_indexes.append(i)            
encoded_start_cat = len(encoded_continuous_feature_indexes)
normalise_weights={}

for idx in encoded_continuous_feature_indexes:
    _max= float(np.max( dataset[:,idx] ))
    _min= float(np.min( dataset[:,idx] ))
    normalise_weights[idx]=[_min, _max]


logging.warning("Normalized weight {}".format(normalise_weights))
logging.warning("Encoded features {}".format(d.encoded_feature_names))
logging.warning("Continuous encoded features {}".format(encoded_continuous_feature_indexes))
logging.warning("Number of columns before processing {}".format(train_data_vae.shape))




#Normlization for conitnuous features
encoded_data= d.normalize_data(encoded_data)
if dataset_name=='adult':
    # Need to rearrange columns such that the Income comes at the last
    cols = list(encoded_data.columns)
    cols = cols[:2] + cols[3:] + [cols[2]]
    encoded_data = encoded_data[cols]
dataset = encoded_data.to_numpy()

#Train, Val, Test Splits
np.random.shuffle(dataset)
test_size= int(0.1*dataset.shape[0])
vae_test_dataset= dataset[:test_size]
dataset= dataset[test_size:]
vae_val_dataset= dataset[:test_size]
vae_train_dataset= dataset[test_size:]

# Saving dataets 
np.save(base_dir+dataset_name+'-'+'train-set', vae_train_dataset)
np.save(base_dir+dataset_name+'-'+'val-set', vae_val_dataset)
np.save(base_dir+dataset_name+'-'+'test-set', vae_test_dataset)

logging.warning("Number of columns before processing {}".format(vae_train_dataset.shape))

#Saving Normalise Weights
f=open(base_dir+dataset_name+'-'+'normalise_weights.json', 'w')
json.dump(normalise_weights, f)
f.close()

#Saving MAD 
f=open(base_dir+dataset_name+'-'+'mad.json', 'w')
json.dump(mad_feature_weights, f)
f.close()
