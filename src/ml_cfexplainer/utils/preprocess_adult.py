import pandas as pd
import yaml

from sklearn.preprocessing import LabelEncoder

from helpers import load_adult_income_dataset
from dataloader import DataLoader

import feather



if __name__ == "__main__":
    """Load configuration"""
    with open('/home/trduong/Data/multiobj-scm-cf/src/config.yml') as file:
        conf = yaml.safe_load(file)

    """Load data and dataloader and normalize data"""
    dataset = load_adult_income_dataset()
    params = {'dataframe': dataset.copy(),
              'continuous_features': ['age', 'hours_per_week'],
              'outcome_name': 'income'}
    d = DataLoader(params)
    df = d.data_df

    df.to_feather(conf['data_adult'])

    df = d.normalize_data(df)

    for v in d.categorical_feature_names:
        le = LabelEncoder()
        df[v] = le.fit_transform(df[v].values)

    for v in d.categorical_feature_names:
        df[v] = pd.Categorical(df[v].values)

    df.to_feather(conf['processed_data_adult'])


