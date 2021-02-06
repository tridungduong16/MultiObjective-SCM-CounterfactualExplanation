"""
This module containts helper functions to load data and get meta deta.
"""
import numpy as np
import pandas as pd
import os

def load_adult_income_dataset(save_intermediate=False):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :param: save_intermediate: save the transformed dataset. Do not save by default.
    """
    raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', delimiter=', ', dtype=str)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)


    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                           'Exec-managerial':'White-Collar','Farming-fishing':'Blue-Collar',
                                            'Handlers-cleaners':'Blue-Collar',
                                            'Machine-op-inspct':'Blue-Collar','Other-service':'Service',
                                            'Priv-house-serv':'Service',
                                           'Prof-specialty':'Professional','Protective-serv':'Service',
                                            'Tech-support':'Service',
                                           'Transport-moving':'Blue-Collar','Unknown':'Other/Unknown',
                                            'Armed-Forces':'Other/Unknown','?':'Other/Unknown'}})

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married','Never-married':'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                           'Amer-Indian-Eskimo':'Other'}})

    adult_data = adult_data[['age','workclass','education','marital-status','occupation','race','gender',
                     'hours-per-week','income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                           '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                          '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    if save_intermediate:
        pass#adult_data.to_csv('adult.csv', index=False)

    return adult_data

def get_adult_income_modelpath():
    pkg_path = dice_ml.__path__[0]
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'adult.h5')
    return modelpath

def get_adult_data_info():
    feature_description = {'age':'age',
                        'workclass': 'type of industry (Government, Other/Unknown, Private, Self-Employed)',
                        'education': 'education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)',
                        'marital_status': 'marital status (Divorced, Married, Separated, Single, Widowed)',
                        'occupation': 'occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)',
                        'race': 'white or other race?',
                        'gender': 'male or female?',
                        'hours_per_week': 'total work hours per week',
                        'income': '0 (<=50K) vs 1 (>50K)'}
    return feature_description
