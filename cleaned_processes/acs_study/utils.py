"""
Utility functions for ACS Census data loading and preprocessing.
Adapted from https://github.com/tijana-zrnic/active-inference
"""

import numpy as np
import scipy
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os


def get_data(year, features, outcome, data_dir="data", randperm=True):
    csv_path = os.path.join(data_dir, str(year), "1-Year", "psam_p06.csv")
    usecols = list(set(features + [outcome, 'COW']))

    if os.path.exists(csv_path):
        acs_data = pd.read_csv(csv_path, usecols=usecols)
    else:
        import folktables
        data_source = folktables.ACSDataSource(
            survey_year=year, horizon='1-Year', survey='person',
            root_dir=data_dir
        )
        acs_data = data_source.get_data(states=["CA"], download=True)

    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    employed = np.isin(acs_data['COW'], np.array([1, 2, 3, 4, 5, 6, 7]))
    if randperm:
        shuffler = np.random.permutation(income.shape[0])
        income_features = income_features.iloc[shuffler]
        income = income.iloc[shuffler]
        employed = employed[shuffler]
    return income_features, income, employed


def transform_features(features, ft, enc=None):
    c_features = features.T[ft == "c"].T.astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)
        enc.fit(c_features)
    c_features = enc.transform(c_features)
    features_enc = scipy.sparse.csc_matrix(
        np.concatenate([features.T[ft == "q"].T.astype(float), c_features], axis=1)
    )
    return features_enc, enc


def ols(features, outcome):
    return np.linalg.pinv(features).dot(outcome)
