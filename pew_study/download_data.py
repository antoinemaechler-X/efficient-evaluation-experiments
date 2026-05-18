"""
Download and preprocess Pew ATP Wave 79 dataset.

Loads the SPSS file (ATPW79.sav), extracts features and binary outcome,
trains XGBoost, and saves prediction arrays.

Follows Zrnic & Candes (2024) exactly:
  - 50/50 train/test split
  - Main model trained on FULL training data → predictions on test
  - Separate tuning model trained on 80% of training → predictions on remaining 20%
  - theta_true = mean(Y_all) (all data, not just test)

Dataset: Pew Research Center ATP Wave 79 (November 2020, post-election).
Estimand: Biden post-election messaging approval rate.

Requires:
  - pyreadstat (pip install pyreadstat)
  - xgboost (pip install xgboost)
  - The ATPW79.sav file from Pew Research Center (free registration required)
    Place at: pew_study/data/ATPW79.sav

Usage: python pew_study/download_data.py
"""
import numpy as np
import os
import sys

try:
    import pyreadstat
except ImportError:
    print("pyreadstat not installed. Run: pip install pyreadstat")
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

from sklearn.model_selection import train_test_split

# --- Paths ---
data_dir = "pew_study/data"
os.makedirs(data_dir, exist_ok=True)

sav_path = os.path.join(data_dir, "ATPW79.sav")
if not os.path.exists(sav_path):
    print(f"ERROR: SPSS file not found at {sav_path}")
    print("Please download ATPW79.sav from Pew Research Center:")
    print("  https://www.pewresearch.org/dataset/american-trends-panel-wave-79/")
    print(f"Then place it at: {sav_path}")
    sys.exit(1)

# --- Load SPSS file ---
print(f"Loading {sav_path}...")
data, meta = pyreadstat.read_sav(sav_path)
print(f"  Total rows: {len(data)}, columns: {len(data.columns)}")

# --- Define features and outcome (same as Zrnic notebook) ---
QUESTION = "ELECTBIDENMSSG_W79"

FEATURES = [
    "F_PARTYSUM_FINAL",
    "COVIDFOL_W79",
    "COVIDTHREAT_a_W79",
    "COVIDTHREAT_b_W79",
    "COVIDTHREAT_c_W79",
    "COVIDTHREAT_d_W79",
    "COVIDMASK1_W79",
    "COVID_SCI6E_W79",
    "F_EDUCCAT",
    "F_AGECAT",
]

# --- Filter and binarize (exactly as Zrnic) ---
# Zrnic: idx_keep = np.where(data[question] != 99)[0]
#         Y_all = data[question].to_numpy()[idx_keep] < 2.5
#         X_all = data[features].to_numpy()[idx_keep]
idx_keep = np.where(data[QUESTION] != 99)[0]
Y_all = (data[QUESTION].to_numpy()[idx_keep] < 2.5).astype(np.float32)
X_all = data[FEATURES].to_numpy()[idx_keep].astype(np.float32)

N_total = len(Y_all)
theta_true = Y_all.mean()  # Population parameter (same as Zrnic)

print(f"\nAfter filtering (excluding refused=99):")
print(f"  N = {N_total}")
print(f"  Approve (Y=1): {int(Y_all.sum())} ({Y_all.mean():.4f})")
print(f"  Disapprove (Y=0): {int(N_total - Y_all.sum())} ({1 - Y_all.mean():.4f})")
print(f"  theta_true = {theta_true:.6f}")

# --- Train/test split (50/50, same as Zrnic) ---
SPLIT_SEED = 42
X_train, X_test, Y_train, Y_test = train_test_split(
    X_all, Y_all, test_size=0.5, random_state=SPLIT_SEED
)

print(f"\nTrain/test split (seed={SPLIT_SEED}):")
print(f"  Train: N={len(Y_train)}, mean(Y)={Y_train.mean():.4f}")
print(f"  Test:  N={len(Y_test)}, mean(Y)={Y_test.mean():.4f}")

# === MODEL 1: Main model on FULL training data → predictions on test ===
print("\nTraining main XGBoost on full training data...")
params = {
    "eta": 0.001,
    "max_depth": 5,
    "objective": "reg:logistic",
    "verbosity": 0,
}
NUM_ROUNDS = 3000

dtrain_full = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)

model_main = xgb.train(params, dtrain_full, NUM_ROUNDS)
Yhat_test = model_main.predict(dtest).astype(np.float32)

print(f"  Yhat_test range: [{Yhat_test.min():.4f}, {Yhat_test.max():.4f}]")
print(f"  Yhat_test mean:  {Yhat_test.mean():.4f} (true test mean: {Y_test.mean():.4f})")
print(f"  MSE:             {((Y_test - Yhat_test)**2).mean():.4f}")
print(f"  Accuracy:        {((Yhat_test > 0.5) == Y_test).mean():.4f}")

# === MODEL 2: Tuning model on 80% of training → predictions on 20% (for tau tuning) ===
# Zrnic: train_test_split(X_train, y_train, test_size=0.2)
print("\nTraining tuning XGBoost on 80% of training data...")
X_train1, X_train2, Y_train1, Y_train2 = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=SPLIT_SEED + 1
)

print(f"  Train1 (model fitting): N={len(Y_train1)}")
print(f"  Train2 (tau tuning):    N={len(Y_train2)}")

dtrain1 = xgb.DMatrix(X_train1, label=Y_train1)
dtrain2 = xgb.DMatrix(X_train2, label=Y_train2)

model_tuning = xgb.train(params, dtrain1, NUM_ROUNDS)
Yhat_train2 = model_tuning.predict(dtrain2).astype(np.float32)

print(f"  Yhat_train2 range: [{Yhat_train2.min():.4f}, {Yhat_train2.max():.4f}]")
print(f"  Yhat_train2 mean:  {Yhat_train2.mean():.4f} (true train2 mean: {Y_train2.mean():.4f})")

# --- Save arrays ---
np.save(os.path.join(data_dir, "Y_test.npy"), Y_test)
np.save(os.path.join(data_dir, "Yhat_test.npy"), Yhat_test)
np.save(os.path.join(data_dir, "Y_train2.npy"), Y_train2)
np.save(os.path.join(data_dir, "Yhat_train2.npy"), Yhat_train2)
np.save(os.path.join(data_dir, "theta_true.npy"), np.array([theta_true], dtype=np.float32))

# --- Summary ---
print(f"\nSaved to {data_dir}/:")
print(f"  Y_test.npy:      shape={Y_test.shape}")
print(f"  Yhat_test.npy:   shape={Yhat_test.shape}")
print(f"  Y_train2.npy:    shape={Y_train2.shape}")
print(f"  Yhat_train2.npy: shape={Yhat_train2.shape}")
print(f"  theta_true.npy:  [{theta_true:.6f}] (mean of ALL data, before split)")

N_test = len(Y_test)
print(f"\n{'='*60}")
print(f"Population: N_all = {N_total}, theta_true = {theta_true:.6f}")
print(f"Test set (inference pool): N_test = {N_test}")
print(f"{'='*60}")
