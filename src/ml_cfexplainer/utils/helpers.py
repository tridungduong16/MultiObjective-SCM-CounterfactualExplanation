"""
This module containts helper functions to load data and get meta deta.
"""
import numpy as np
import pandas as pd
import os
import yaml

def load_adult_income_dataset(path, result_path, save_intermediate=False):
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :param: save_intermediate: save the transformed dataset. Do not save by default.
    """
    adult_data = pd.read_csv(path, delimiter = ", ", engine='python')
    # print(df)

    # raw_data = np.genfromtxt(path, delimiter=', ', dtype=str)
    # sys.exit(1)

    # data = np.genfromtxt(path_to_csv, dtype=float, delimiter=',', names=True)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    # column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    # print(adult_data)

    # adult_data = pd.DataFrame(data=raw_data[1:], columns=raw_data[0])

    # print(adult_data)

    # sys.exit(1)
    print(adult_data)
    adult_data = adult_data.rename(columns={'marital-status': 'marital_status',
                                            'hours-per-week': 'hours_per_week',
                                            'sex':'gender'})
    # print(adult_data.columns)

    adult_data[['age', 'hours_per_week']] = adult_data[['age', 'hours_per_week']].astype(int)
    adult_data['income'] = adult_data['income'].astype(str)

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

    adult_data = adult_data.replace({'marital_status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married','Never-married':'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                           'Amer-Indian-Eskimo':'Other'}})

    adult_data = adult_data[['age','workclass','education','marital_status','occupation','race','gender',
                     'hours_per_week','income']]

    adult_data['income'] = np.where(adult_data.income == '<=50K', 0, 1)


    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                           '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                          '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})



    if save_intermediate:
        adult_data.to_csv(result_path, index=False)

    return adult_data


if __name__ == "__main__":
    """Load configuration"""
    with open("/home/trduong/Data/counterfactual_fairness_game_theoric/configuration.yml", 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    df = load_adult_income_dataset(conf['data_adult'], conf['processed_data_adult'], True)
