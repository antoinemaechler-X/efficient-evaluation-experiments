"""
Download and preprocess Amazon deforestation dataset.

Downloads data.csv from the cross-ppi GitHub repo, preprocesses exactly as
Zrnic & Candes (2024), trains HistGradientBoostingClassifier, saves arrays.

Two usage modes:
  - Our methods + Active Inference: 50/50 train/test split, model on training half
  - Cross-PPI: operates on FULL dataset with K-fold cross-fitting (needs X_all, Y_all)

Dataset: Bullock et al. (2020) Amazon deforestation labels + Sexton et al. (2013) canopy cover.
Estimand: fraction of parcels deforested between 2000-2015.

Source: https://github.com/tijana-zrnic/cross-ppi/blob/main/deforestation/data.csv

Usage: python deforestation_study/download_data.py
"""
import numpy as np
import pandas as pd
import os
import sys
import urllib.request

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

# --- Paths ---
data_dir = "deforestation_study/data"
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "data.csv")

# --- Download data.csv from cross-ppi repo ---
if not os.path.exists(csv_path):
    url = "https://raw.githubusercontent.com/tijana-zrnic/cross-ppi/main/deforestation/data.csv"
    print(f"Downloading data.csv from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    print(f"  Saved to {csv_path}")
else:
    print(f"Data already exists at {csv_path}")

# --- Preprocessing: EXACTLY as Zrnic's deforestation.ipynb ---
raw_df = pd.read_csv(csv_path)
print(f"Raw data: {len(raw_df)} rows, {len(raw_df.columns)} columns")

# Filter bad row (Year1 = 'tdo')
df = raw_df[raw_df.Year1 != 'tdo'].copy()
df.Year1 = df.Year1.astype(float)

# Y: binary deforestation indicator (any disturbance event between 2000-2015)
Y_all = (
    ((df.Type1.astype(str) != 'nan') & ((df.Year1 >= 2000) & (df.Year1 <= 2015))) |
    ((df.Type2.astype(str) != 'nan') & ((df.Year2 >= 2000) & (df.Year2 <= 2015))) |
    ((df.Type3.astype(str) != 'nan') & ((df.Year3 >= 2000) & (df.Year3 <= 2015)))
).astype(float).to_numpy()

# X: 2D features (tree canopy cover in 2015 and 2000, scaled to [0,1])
X_all = np.stack([
    df.tree_canopy_cover_2015 / 100,
    df.tree_canopy_cover_2000 / 100
], axis=1)

N_total = len(Y_all)
theta_true = Y_all.mean()

print(f"\nAfter filtering (excluding 'tdo' row):")
print(f"  N = {N_total}")
print(f"  Deforested (Y=1): {int(Y_all.sum())} ({Y_all.mean():.4f})")
print(f"  Not deforested (Y=0): {int(N_total - Y_all.sum())} ({1 - Y_all.mean():.4f})")
print(f"  theta_true = {theta_true:.6f}")
print(f"  X shape: {X_all.shape}, features: canopy_2015, canopy_2000")

# --- Save full dataset arrays (needed for Cross-PPI cross-fitting) ---
np.save(os.path.join(data_dir, "X_all.npy"), X_all.astype(np.float32))
np.save(os.path.join(data_dir, "Y_all.npy"), Y_all.astype(np.float32))

# --- Train/test split (50/50) for our methods + Active Inference ---
SPLIT_SEED = 42
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y_all, test_size=0.5, random_state=SPLIT_SEED
)

print(f"\nTrain/test split (seed={SPLIT_SEED}):")
print(f"  Train: N={len(Y_train)}, mean(Y)={Y_train.mean():.4f}")
print(f"  Test:  N={len(Y_test)}, mean(Y)={Y_test.mean():.4f}")

# === MODEL 1: Main model on FULL training data → predictions on test ===
# Uses same HistGradientBoostingClassifier as Zrnic's deforestation notebook
print("\nTraining main HistGradientBoostingClassifier on full training data...")
cls_main = HistGradientBoostingClassifier(max_iter=100, max_depth=2)
cls_main.fit(X_train, Y_train)
Yhat_test = cls_main.predict_proba(X_test)[:, 1].astype(np.float32)

print(f"  Yhat_test range: [{Yhat_test.min():.4f}, {Yhat_test.max():.4f}]")
print(f"  Yhat_test mean:  {Yhat_test.mean():.4f} (true test mean: {Y_test.mean():.4f})")
print(f"  MSE:             {((Y_test - Yhat_test)**2).mean():.4f}")
print(f"  Accuracy:        {((Yhat_test > 0.5) == Y_test).mean():.4f}")

# === MODEL 2: Tuning model on 80% of training → predictions on 20% ===
# Used only by Active Inference for tau tuning (same pattern as pew_study)
print("\nTraining tuning HistGradientBoostingClassifier on 80% of training data...")
X_train1, X_train2, Y_train1, Y_train2 = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=SPLIT_SEED + 1
)

print(f"  Train1 (model fitting): N={len(Y_train1)}")
print(f"  Train2 (tau tuning):    N={len(Y_train2)}")

cls_tuning = HistGradientBoostingClassifier(max_iter=100, max_depth=2)
cls_tuning.fit(X_train1, Y_train1)
Yhat_train2 = cls_tuning.predict_proba(X_train2)[:, 1].astype(np.float32)

print(f"  Yhat_train2 range: [{Yhat_train2.min():.4f}, {Yhat_train2.max():.4f}]")
print(f"  Yhat_train2 mean:  {Yhat_train2.mean():.4f} (true train2 mean: {Y_train2.mean():.4f})")

# --- Save arrays ---
np.save(os.path.join(data_dir, "Y_test.npy"), Y_test.astype(np.float32))
np.save(os.path.join(data_dir, "Yhat_test.npy"), Yhat_test)
np.save(os.path.join(data_dir, "Y_train2.npy"), Y_train2.astype(np.float32))
np.save(os.path.join(data_dir, "Yhat_train2.npy"), Yhat_train2)
np.save(os.path.join(data_dir, "theta_true.npy"), np.array([theta_true], dtype=np.float32))

# --- Summary ---
print(f"\nSaved to {data_dir}/:")
print(f"  X_all.npy:       shape={X_all.shape}  (for Cross-PPI cross-fitting)")
print(f"  Y_all.npy:       shape={Y_all.shape}  (for Cross-PPI cross-fitting)")
print(f"  Y_test.npy:      shape={Y_test.shape}")
print(f"  Yhat_test.npy:   shape={Yhat_test.shape}")
print(f"  Y_train2.npy:    shape={Y_train2.shape}")
print(f"  Yhat_train2.npy: shape={Yhat_train2.shape}")
print(f"  theta_true.npy:  [{theta_true:.6f}] (mean of ALL data, before split)")

N_test = len(Y_test)
print(f"\n{'='*60}")
print(f"Population: N_all = {N_total}, theta_true = {theta_true:.6f}")
print(f"Test set (inference pool): N_test = {N_test}")
print(f"Full dataset (Cross-PPI pool): N_all = {N_total}")
print(f"{'='*60}")
