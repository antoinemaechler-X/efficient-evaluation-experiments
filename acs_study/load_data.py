"""
Test script: loads ACS Census data for California 2019 and prints basic stats.
Run this to verify the data pipeline works.
"""

import numpy as np
from utils import get_data, transform_features, ols

# --- Feature definitions ---
features = [
    'AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
    'ANC1P', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P', 'SOCP', 'COW'
]
# q = quantitative, c = categorical
ft = np.array([
    "q", "q", "c", "c", "c", "c", "c", "c",
    "c", "c", "c", "c", "c", "c", "c", "c", "c"
])

# --- Load data ---
print("Loading ACS PUMS data for CA 2019...")
income_features, income, employed = get_data(year=2019, features=features, outcome='PINCP')

N = len(income)
d = len(features)
print(f"Loaded {N} individuals with {d} features.")
print(f"Employed: {employed.sum()} / {N} ({100*employed.mean():.1f}%)")
print(f"Income: mean={income.mean():.2f}, median={income.median():.2f}, "
      f"min={income.min()}, max={income.max()}")

# --- Transform features ---
print("\nEncoding features (one-hot for categoricals)...")
income_features_enc, enc = transform_features(income_features, ft)
print(f"Encoded feature matrix shape: {income_features_enc.shape}")

# --- Quick OLS sanity check ---
age = income_features['AGEP'].to_numpy()
sex = income_features['SEX'].to_numpy()
X = np.stack([age, sex], axis=1)
theta = ols(X, income.to_numpy())
print(f"\nOLS coefficients (age, sex) on income: {theta}")

print("\nData loading OK.")
