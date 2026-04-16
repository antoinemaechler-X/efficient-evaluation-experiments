"""
Train XGBoost model to predict personal income (PINCP) from ACS demographic features.
Mirrors the setup in active-inference/census-analysis.ipynb.

Produces:
  - predictions Yhat for the unlabeled split
  - predicted |error| estimates (uncertainty proxy)
  - ground-truth Y on the unlabeled split
  - the covariate matrix X (age, sex) used for the downstream regression target

Saves everything to <out> (a .npz file) for reuse by FAQ/PAI experiments.

Usage:
    python train_model.py                  # quick test (200 rounds)
    python train_model.py --n_rounds 2000  # full, matches paper
    python train_model.py --seed 0 --n_rounds 500
"""

import argparse
import os
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
import xgboost as xgb

from utils import get_data, transform_features, ols


FEATURES = [
    'AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
    'ANC1P', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P', 'SOCP', 'COW'
]
FT = np.array([
    "q", "q", "c", "c", "c", "c", "c", "c",
    "c", "c", "c", "c", "c", "c", "c", "c", "c"
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2019)
    parser.add_argument('--n_rounds', type=int, default=200,
                        help="XGBoost boosting rounds (paper uses 2000).")
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--max_depth', type=int, default=7)
    parser.add_argument('--train_frac', type=float, default=0.5,
                        help="Fraction of data used for training the ML model.")
    parser.add_argument('--n_labeled', type=int, default=None,
                        help="Absolute number of labeled points for training. "
                             "If set, overrides --train_frac.")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='predictions.npz')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Load ACS data ---
    print(f"Loading ACS {args.year} CA data...")
    income_features, income, employed = get_data(
        year=args.year, features=FEATURES, outcome='PINCP'
    )
    n_all = len(income)
    if args.n_labeled is not None:
        n_tr = args.n_labeled
    else:
        n_tr = int(n_all * args.train_frac)
    print(f"N = {n_all}, train = {n_tr}, unlabeled = {n_all - n_tr}")

    # --- Split into labeled / unlabeled ---
    (feats_labeled, feats_unlabeled,
     income_labeled, income_unlabeled) = train_test_split(
        income_features, income, train_size=n_tr, random_state=args.seed
    )
    income_labeled = income_labeled.to_numpy()
    Y = income_unlabeled.to_numpy()

    # Covariates for the regression target (age, sex) on the unlabeled side.
    # The estimand is the OLS coefficient on AGEP when regressing income on (age, sex).
    age = income_features['AGEP'].to_numpy()
    sex = income_features['SEX'].to_numpy()
    theta_true = ols(np.stack([age, sex], axis=1), income.to_numpy())[0]
    print(f"Ground-truth theta (OLS coeff on AGEP): {theta_true:.4f}")

    X = np.stack([
        feats_unlabeled['AGEP'].to_numpy(),
        feats_unlabeled['SEX'].to_numpy()
    ], axis=1)

    # --- Encode features ---
    print("Encoding features...")
    _, enc = transform_features(income_features, FT)
    feats_labeled_enc, _ = transform_features(feats_labeled, FT, enc)
    feats_unlabeled_enc, _ = transform_features(feats_unlabeled, FT, enc)

    # --- Train primary income predictor (median regression) ---
    print(f"Training income model ({args.n_rounds} rounds)...")
    dtrain = xgb.DMatrix(feats_labeled_enc, label=income_labeled)
    tree = xgb.train(
        {'eta': args.eta, 'max_depth': args.max_depth,
         'objective': 'reg:absoluteerror'},
        dtrain, args.n_rounds
    )
    Yhat = tree.predict(xgb.DMatrix(feats_unlabeled_enc))

    # --- Train auxiliary |error| model for uncertainty ---
    print(f"Training error-magnitude model ({args.n_rounds} rounds)...")
    train_preds = tree.predict(xgb.DMatrix(feats_labeled_enc))
    dtrain_err = xgb.DMatrix(
        feats_labeled_enc,
        label=np.abs(income_labeled - train_preds)
    )
    tree_err = xgb.train(
        {'eta': args.eta, 'max_depth': args.max_depth,
         'objective': 'reg:absoluteerror'},
        dtrain_err, args.n_rounds
    )
    predicted_errs = np.clip(
        tree_err.predict(xgb.DMatrix(feats_unlabeled_enc)), 0, np.inf
    )

    # --- Quick quality check ---
    mae = np.mean(np.abs(Y - Yhat))
    print(f"Test MAE on unlabeled split: {mae:.2f}")
    print(f"Predicted-error mean/median/max: "
          f"{predicted_errs.mean():.2f} / "
          f"{np.median(predicted_errs):.2f} / "
          f"{predicted_errs.max():.2f}")

    # --- Save for downstream FAQ/PAI experiments ---
    out_path = os.path.abspath(args.out)
    np.savez_compressed(
        out_path,
        Y=Y, Yhat=Yhat, predicted_errs=predicted_errs,
        X=X, theta_true=theta_true,
    )
    print(f"Saved predictions to {out_path}")

    # Companion files for the sequential PAI/FAQ experiment:
    #   <out_stem>.tree.json        — booster for income prediction
    #   <out_stem>.tree_err.json    — booster for |error| prediction
    #   <out_stem>.features.npz     — encoded unlabeled feature matrix (sparse)
    stem, _ = os.path.splitext(out_path)
    tree.save_model(stem + '.tree.json')
    tree_err.save_model(stem + '.tree_err.json')
    scipy.sparse.save_npz(stem + '.features.npz', feats_unlabeled_enc)
    print(f"Saved XGBoost models to {stem}.tree.json, {stem}.tree_err.json")
    print(f"Saved encoded unlabeled features to {stem}.features.npz "
          f"(shape {feats_unlabeled_enc.shape}, nnz {feats_unlabeled_enc.nnz})")


if __name__ == '__main__':
    main()
