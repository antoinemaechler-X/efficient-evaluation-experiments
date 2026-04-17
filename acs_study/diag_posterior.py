"""
Diagnostic: does V normalization actually matter for the BLR posterior?

Prints vtSigmav/sigma2 with and without column normalization of V,
for several n_labeled values. This ratio must be ~1 for FAQ to have signal.

Usage:
    python diag_posterior.py
"""

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from sklearn.model_selection import train_test_split
from utils import get_data, transform_features

FEATURES = [
    'AGEP', 'SCHL', 'MAR', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
    'ANC1P', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P', 'SOCP', 'COW'
]
FT = np.array([
    "q", "q", "c", "c", "c", "c", "c", "c",
    "c", "c", "c", "c", "c", "c", "c", "c", "c"
])
D = 16

print("Loading data...")
income_features, income, _ = get_data(year=2019, features=FEATURES, outcome='PINCP')
n_all = len(income)
n_tr  = int(n_all * 0.5)

feats_lab, feats_unlab, y_lab, y_unlab = train_test_split(
    income_features, income, train_size=n_tr, random_state=0
)
y_lab = y_lab.to_numpy()

_, enc = transform_features(income_features, FT)
Phi_lab,   _ = transform_features(feats_lab,   FT, enc)
Phi_unlab, _ = transform_features(feats_unlab, FT, enc)

print(f"Running SVD (D={D})...")
Phi_all = scipy.sparse.vstack([Phi_lab, Phi_unlab]).tocsr().astype(np.float32)
U_svd, s_svd, _ = scipy.sparse.linalg.svds(Phi_all, k=D)
order = np.argsort(-s_svd)
s_svd = s_svd[order]; U_svd = U_svd[:, order]

V_raw = (U_svd * s_svd).astype(np.float64)             # un-normalized
V_col_std = V_raw.std(axis=0, keepdims=True).clip(1e-8)
V_norm = V_raw / V_col_std                              # normalized

print(f"\nSingular values (top {D}): {s_svd.round(1)}")
print(f"V_raw  col std: min={V_raw.std(axis=0).min():.3f}  max={V_raw.std(axis=0).max():.3f}")
print(f"V_norm col std: min={V_norm.std(axis=0).min():.3f}  max={V_norm.std(axis=0).max():.3f}")
print(f"V_raw  row norm: mean={np.linalg.norm(V_raw,  axis=1).mean():.4f}")
print(f"V_norm row norm: mean={np.linalg.norm(V_norm, axis=1).mean():.4f}")

mu_Y, sd_Y = y_lab.mean(), y_lab.std()
y_lab_n = ((y_lab - mu_Y) / sd_Y).astype(np.float64)


def compute_ratio(V_lab, V_unlab, n_labeled, prior_tau=10.0):
    """Fit BLR on n_labeled points, return mean(v^T Sigma v) / sigma2."""
    n, D = V_lab.shape
    Phi = V_lab[:n_labeled]
    y   = y_lab_n[:n_labeled]

    beta_ols = np.linalg.pinv(Phi) @ y
    sigma2   = float(np.var(y - Phi @ beta_ols))

    prior_prec = np.eye(D) / (prior_tau ** 2)
    post_prec  = prior_prec + (Phi.T @ Phi) / sigma2
    Sig_post   = np.linalg.inv(post_prec)

    # vtSigmav for all unlabeled points (sample 5000 for speed)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(V_unlab), size=min(5000, len(V_unlab)), replace=False)
    V_sample = V_unlab[idx]
    vtSv = np.einsum('nd,de,ne->n', V_sample, Sig_post, V_sample)
    return vtSv.mean() / sigma2, sigma2


print("\n" + "="*65)
print(f"{'n_labeled':>12}  {'V_raw ratio':>14}  {'V_norm ratio':>14}  {'sigma2':>8}")
print("="*65)
for n_lab in [50, 100, 200, 500, 2000, n_tr]:
    r_raw,  s2_raw  = compute_ratio(V_raw[:n_tr],  V_raw[n_tr:],  n_lab)
    r_norm, s2_norm = compute_ratio(V_norm[:n_tr], V_norm[n_tr:], n_lab)
    print(f"{n_lab:>12,}  {r_raw:>14.6f}  {r_norm:>14.6f}  {s2_raw:>8.4f}")

print("="*65)
print("\nInterpretation:")
print("  ratio << 1  →  posterior collapsed  →  h_o uniform  →  FAQ = uniform+pai")
print("  ratio  ~1   →  posterior uncertain  →  h_o non-uniform  →  FAQ has signal")
print("  ratio >> 1  →  prior dominates      →  posterior not updated yet")
