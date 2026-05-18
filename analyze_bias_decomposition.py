"""
Analyze verify_bias_decomposition results.
Two claims to verify:
  1) T_1 is the dominant term in B̂_n - B_n = T_1 + T_2 - T_3
  2) B_s ≪ A_s at every step (justifies v_s estimation)
"""

import os, glob
import numpy as np
import pandas as pd

LOG_DIR = "logs/verify_bias"

# ── Collect all decomposition files ──────────────────────────────────────
dec_files = sorted(glob.glob(os.path.join(LOG_DIR, "decomposition_*.csv")))
prof_files = sorted(glob.glob(os.path.join(LOG_DIR, "profiles_*.csv")))

print("=" * 72)
print("CLAIM 1: T_1 dominates the decomposition B̂_n - B_n = T_1 + T_2 - T_3")
print("=" * 72)

rows = []
for f in dec_files:
    # Parse dataset and budget from filename
    base = os.path.basename(f).replace("decomposition_", "").replace(".csv", "")
    parts = base.rsplit("_", 1)
    dataset, budget = parts[0], float(parts[1])

    df = pd.read_csv(f)
    n = len(df)

    T1 = df["T1"].mean()
    T2 = df["T2"].mean()
    T3 = df["T3"].mean()
    Bhat_B = df["Bhat_minus_Bn"].mean()
    sigma2 = df["sigma_bar_sq"].mean()

    # Sanity check: B̂-B should ≈ T1 + T2 - T3
    recon = T1 + T2 - T3
    sanity_err = abs(Bhat_B - recon) / (abs(Bhat_B) + 1e-30)

    rows.append({
        "dataset": dataset, "budget": budget, "n_seeds": n,
        "E[T1]": T1, "E[T2]": T2, "T3": T3,
        "E[B̂-B]": Bhat_B, "recon_err": sanity_err,
        "|T1|/|B̂-B|": abs(T1) / (abs(Bhat_B) + 1e-30),
        "|T2|/|T1|": abs(T2) / (abs(T1) + 1e-30),
        "T3/|T1|": abs(T3) / (abs(T1) + 1e-30),
        "E[T1]/σ̄²": T1 / (sigma2 + 1e-30),
        "E[T2]/σ̄²": T2 / (sigma2 + 1e-30),
    })

dec_summary = pd.DataFrame(rows).sort_values(["dataset", "budget"])

# Print nicely
for dataset in dec_summary["dataset"].unique():
    sub = dec_summary[dec_summary["dataset"] == dataset]
    print(f"\n── {dataset} ──")
    print(f"{'budget':>7s}  {'E[T1]':>11s}  {'E[T2]':>11s}  {'T3':>11s}  "
          f"{'|T2|/|T1|':>9s}  {'T3/|T1|':>9s}  {'recon_err':>9s}  "
          f"{'T1/σ̄²':>8s}  {'T2/σ̄²':>8s}")
    for _, r in sub.iterrows():
        print(f"{r['budget']:7.3f}  {r['E[T1]']:11.2e}  {r['E[T2]']:11.2e}  {r['T3']:11.2e}  "
              f"{r['|T2|/|T1|']:9.4f}  {r['T3/|T1|']:9.4f}  {r['recon_err']:9.2e}  "
              f"{r['E[T1]/σ̄²']:8.4f}  {r['E[T2]/σ̄²']:8.4f}")

print("\n")
print("=" * 72)
print("CLAIM 2: B_s ≪ A_s at every step (B_s/A_s ratio)")
print("=" * 72)

for f in prof_files:
    base = os.path.basename(f).replace("profiles_", "").replace(".csv", "")
    parts = base.rsplit("_", 1)
    dataset, budget = parts[0], float(parts[1])

    df = pd.read_csv(f)
    ratio = df["B_over_A"]
    n_steps = len(df)

    # Statistics on B/A ratio
    print(f"\n── {dataset}  budget={budget:.3f}  ({n_steps} steps) ──")
    print(f"  B_s/A_s:  mean={ratio.mean():.5f}  max={ratio.max():.5f}  "
          f"min={ratio.min():.5f}  final={ratio.iloc[-1]:.5f}")

    # Show profile at a few checkpoints
    checkpoints = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    print(f"  {'step':>6s}  {'A_s':>11s}  {'B_s':>11s}  {'B/A':>8s}")
    for idx in checkpoints:
        r = df.iloc[idx]
        print(f"  {int(r['step']):6d}  {r['A_s']:11.4e}  {r['B_s']:11.4e}  {r['B_over_A']:8.5f}")
