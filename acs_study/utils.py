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
    """Load ACS PUMS data for California.

    Reads only the needed columns from the CSV to minimize memory usage.
    Expects the CSV at <data_dir>/<year>/1-Year/psam_p06.csv.
    If the file doesn't exist, falls back to folktables to download it.

    Args:
        year: Survey year (e.g. 2019).
        features: List of feature column names.
        outcome: Outcome column name (e.g. 'PINCP' for personal income).
        data_dir: Root directory for cached data.
        randperm: Whether to randomly permute the rows.

    Returns:
        income_features: DataFrame of covariates.
        income: Series of outcome values.
        employed: Boolean array indicating employment status.
    """
    csv_path = os.path.join(data_dir, str(year), "1-Year", "psam_p06.csv")

    # Columns we actually need (features + outcome + COW for employment)
    usecols = list(set(features + [outcome, 'COW']))

    if os.path.exists(csv_path):
        acs_data = pd.read_csv(csv_path, usecols=usecols)
    else:
        # Fall back to folktables download
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
    """One-hot encode categorical features and concatenate with quantitative ones.

    Args:
        features: DataFrame or array of raw features.
        ft: Array of feature types ('c' = categorical, 'q' = quantitative).
        enc: Optional pre-fitted OneHotEncoder.

    Returns:
        features_enc: Sparse CSC matrix of encoded features.
        enc: The fitted OneHotEncoder.
    """
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
    """Ordinary least squares via pseudo-inverse."""
    return np.linalg.pinv(features).dot(outcome)
