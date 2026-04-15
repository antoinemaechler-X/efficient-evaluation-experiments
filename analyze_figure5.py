'''
Task 3: Full Figure 5 — CI-width ratio (ablation / FAQ) across 2.5%-50% budgets.
All missingness settings.

Combines:
  - Low budgets (2.5%-25%): original paper results from faq_final_logs.csv
    and active_inference_ablation_logs.csv
  - High budgets (25%-50%) fully-obs: faq_final_high_budget_sl=*.csv
  - High budgets (25%-50%) ms=1: faq_final_hb_allms_sl=*.csv
  - Ablation high budgets: active_inference_ablation_dataset=*.csv

Produces:
  - figures/figure5_full_range.png  (fully-observed only, main figure)
  - figures/figure5_by_missingness.png  (all missingness on one plot per dataset)

Usage: python analyze_figure5.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]

# ==================================================================
# LOW BUDGETS (2.5%-25%) — original paper results (all missingness)
# ==================================================================

# ---- FAQ low budget ----
faq_low = pd.read_csv("logs/final/faq_final_logs.csv")
faq_low["prop_budget"] = np.round(faq_low["prop_budget"], decimals=3)
print(f"FAQ low-budget (all ms): {len(faq_low)} rows")

# ---- Ablation low budget ----
abl_low = pd.read_csv("logs/final/active_inference_ablation_logs.csv")
abl_low["prop_budget"] = np.round(abl_low["prop_budget"], decimals=3)
print(f"Ablation low-budget (all ms): {len(abl_low)} rows")

# post-hoc best tau per (dataset, missingness, budget) for low budgets
abl_low_means = abl_low.groupby(["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"]).agg(
    mean_width=("mean_width", "mean")
).reset_index()
best_idx = abl_low_means.groupby(["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"])["mean_width"].idxmin()
abl_low_best = abl_low_means.loc[best_idx][["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"]]

best_abl_low_per_seed = pd.merge(
    abl_low, abl_low_best, on=["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"], how="inner"
)
print(f"Ablation low-budget per-seed (post-hoc best): {len(best_abl_low_per_seed)} rows")

# ==================================================================
# HIGH BUDGETS (25%-50%) — new results
# ==================================================================

# ---- FAQ high budget (fully-obs) ----
faq_high_dfs = []
for sl in range(3):
    fname = f"logs/final/faq_final_high_budget_sl={sl}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        # these files have no missingness columns — they are fully-observed
        if "n_full_obs" not in df.columns:
            df["n_full_obs"] = None
            df["mcar_obs_prob"] = 1.0
        faq_high_dfs.append(df)
        print(f"Loaded {fname}: {len(df)} rows")
    else:
        print(f"MISSING: {fname}")

# ---- FAQ high budget (ms=1 missingness settings) ----
for sl in range(3):
    fname = f"logs/final/faq_final_hb_allms_sl={sl}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        faq_high_dfs.append(df)
        print(f"Loaded {fname}: {len(df)} rows")
    else:
        print(f"MISSING: {fname}")

if faq_high_dfs:
    faq_high = pd.concat(faq_high_dfs, ignore_index=True)
    faq_high["prop_budget"] = np.round(faq_high["prop_budget"], decimals=3)
    print(f"FAQ high-budget total: {len(faq_high)} rows")
    print(f"  Missingness settings: {faq_high.groupby(['n_full_obs', 'mcar_obs_prob']).size().to_dict()}")
else:
    faq_high = pd.DataFrame()

# ---- Ablation high budget (all missingness) ----
abl_high_dfs = []
for dataset in DATASETS:
    fname = f"logs/final/active_inference_ablation_dataset={dataset}.csv"
    if os.path.exists(fname):
        abl_high_dfs.append(pd.read_csv(fname))
        print(f"Loaded {fname}: {len(abl_high_dfs[-1])} rows")
    else:
        print(f"MISSING: {fname}")

if abl_high_dfs:
    abl_high = pd.concat(abl_high_dfs, ignore_index=True)
    abl_high["prop_budget"] = np.round(abl_high["prop_budget"], decimals=3)
    # only keep >25% budgets
    abl_high = abl_high[abl_high["prop_budget"] > 0.25]
    print(f"Ablation high-budget (>25%, all ms): {len(abl_high)} rows")

    # post-hoc best tau per (dataset, missingness, budget)
    abl_high_means = abl_high.groupby(["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"]).agg(
        mean_width=("mean_width", "mean")
    ).reset_index()
    best_idx_h = abl_high_means.groupby(["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"])["mean_width"].idxmin()
    abl_high_best = abl_high_means.loc[best_idx_h][["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"]]

    best_abl_high_per_seed = pd.merge(
        abl_high, abl_high_best, on=["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"], how="inner"
    )
    print(f"Ablation high-budget per-seed (post-hoc best): {len(best_abl_high_per_seed)} rows")
else:
    best_abl_high_per_seed = pd.DataFrame()


# ==================================================================
# HELPER: compute ratio summary for a given missingness setting
# ==================================================================
def compute_ratio_summary(faq_low_df, abl_low_df, faq_high_df, abl_high_df, dataset, n_full_obs, mcar_obs_prob):
    """Compute per-budget mean ratio (ablation/FAQ) for a specific (dataset, missingness)."""
    all_summaries = []

    # filter by missingness
    def filter_ms(df):
        if mcar_obs_prob == 1.0:
            return df[(df["dataset"] == dataset) & (df["mcar_obs_prob"] == 1.0)]
        else:
            return df[(df["dataset"] == dataset) &
                       (df["n_full_obs"] == n_full_obs) &
                       (np.isclose(df["mcar_obs_prob"], mcar_obs_prob))]

    # --- Low budgets ---
    faq_sub = filter_ms(faq_low_df)
    abl_sub = filter_ms(abl_low_df)
    if len(faq_sub) > 0 and len(abl_sub) > 0:
        merged = pd.merge(
            faq_sub[["prop_budget", "seed", "mean_width"]].rename(columns={"mean_width": "faq_width"}),
            abl_sub[["prop_budget", "seed", "mean_width"]].rename(columns={"mean_width": "abl_width"}),
            on=["prop_budget", "seed"], how="inner"
        )
        if len(merged) > 0:
            merged["ratio"] = merged["abl_width"] / merged["faq_width"]
            s = merged.groupby("prop_budget").agg(
                mean_ratio=("ratio", "mean"),
                se_ratio=("ratio", lambda x: x.std() / np.sqrt(len(x))),
                n_seeds=("ratio", "count")
            ).reset_index()
            s["range"] = "low"
            all_summaries.append(s)

    # --- High budgets ---
    if len(faq_high_df) > 0 and len(abl_high_df) > 0:
        faq_sub_h = filter_ms(faq_high_df)
        abl_sub_h = filter_ms(abl_high_df)
        if len(faq_sub_h) > 0 and len(abl_sub_h) > 0:
            merged_h = pd.merge(
                faq_sub_h[["prop_budget", "seed", "mean_width"]].rename(columns={"mean_width": "faq_width"}),
                abl_sub_h[["prop_budget", "seed", "mean_width"]].rename(columns={"mean_width": "abl_width"}),
                on=["prop_budget", "seed"], how="inner"
            )
            if len(merged_h) > 0:
                merged_h["ratio"] = merged_h["abl_width"] / merged_h["faq_width"]
                s_h = merged_h.groupby("prop_budget").agg(
                    mean_ratio=("ratio", "mean"),
                    se_ratio=("ratio", lambda x: x.std() / np.sqrt(len(x))),
                    n_seeds=("ratio", "count")
                ).reset_index()
                s_h["range"] = "high"
                all_summaries.append(s_h)

    if not all_summaries:
        return None
    return pd.concat(all_summaries).sort_values("prop_budget")


# ==================================================================
# PLOT 1: Fully-observed only (main Figure 5)
# ==================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, dataset in enumerate(DATASETS):
    ax = axes[i]
    summary = compute_ratio_summary(faq_low, best_abl_low_per_seed,
                                     faq_high, best_abl_high_per_seed,
                                     dataset, None, 1.0)
    if summary is None:
        ax.set_title(f"{dataset}\nNO DATA")
        continue

    ax.errorbar(summary["prop_budget"], summary["mean_ratio"],
                yerr=1.96 * summary["se_ratio"], fmt="o-", capsize=3,
                color="darkblue", label="100% observed")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ratio = 1 (equal)")
    ax.axvline(0.25, color="orange", linestyle=":", alpha=0.5, label="Original paper boundary")
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("CI-Width Ratio (Ablation / FAQ)", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    print(f"\n--- {dataset} (fully-observed) ---")
    for _, row in summary.iterrows():
        marker = "FAQ wins" if row["mean_ratio"] > 1 else "Ablation wins"
        tag = " [NEW]" if row["range"] == "high" else ""
        print(f"  budget={row['prop_budget']:.3f}  |  ratio={row['mean_ratio']:.4f} ± {row['se_ratio']:.4f}  |  {marker}  |  n={int(row['n_seeds'])}{tag}")

plt.suptitle("Figure 5 (full range): CI-Width Ratio — Post-hoc Best Ablation / FAQ", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/figure5_full_range.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: figures/figure5_full_range.png")


# ==================================================================
# PLOT 2: All missingness settings on one plot per dataset
# ==================================================================
missingness_labels = {
    1.0: "Fully observed",
    0.01: "MCAR p=0.01",
    0.001: "MCAR p=0.001",
    0.0001: "MCAR p=0.0001",
    0.00001: "MCAR p=0.00001",
}
ms_colors = {1.0: "darkblue", 0.01: "steelblue", 0.001: "darkorange", 0.0001: "firebrick", 0.00001: "seagreen"}

# collect all unique missingness settings from the data
all_ms = set()
for df in [faq_low, abl_low]:
    for _, row in df[["n_full_obs", "mcar_obs_prob"]].drop_duplicates().iterrows():
        all_ms.add((row["n_full_obs"], row["mcar_obs_prob"]))
all_ms = sorted(all_ms, key=lambda x: -x[1])  # sort by mcar_obs_prob descending

print(f"\nAll missingness settings found: {all_ms}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, dataset in enumerate(DATASETS):
    ax = axes[i]

    for n_full_obs, mcar_obs_prob in all_ms:
        summary = compute_ratio_summary(faq_low, best_abl_low_per_seed,
                                         faq_high, best_abl_high_per_seed,
                                         dataset, n_full_obs, mcar_obs_prob)
        if summary is None:
            continue

        label = missingness_labels.get(mcar_obs_prob, f"nfo={n_full_obs}, p={mcar_obs_prob}")
        color = ms_colors.get(mcar_obs_prob, "gray")
        ax.plot(summary["prop_budget"], summary["mean_ratio"], "o-",
                color=color, markersize=4, label=label)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(0.25, color="orange", linestyle=":", alpha=0.5, label="Paper boundary")
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("CI-Width Ratio (Ablation / FAQ)", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # print summary table
    print(f"\n--- {dataset} (all missingness) ---")
    for n_full_obs, mcar_obs_prob in all_ms:
        summary = compute_ratio_summary(faq_low, best_abl_low_per_seed,
                                         faq_high, best_abl_high_per_seed,
                                         dataset, n_full_obs, mcar_obs_prob)
        if summary is None:
            continue
        lbl = missingness_labels.get(mcar_obs_prob, f"nfo={n_full_obs}, p={mcar_obs_prob}")
        print(f"  {lbl}:")
        for _, row in summary.iterrows():
            marker = "FAQ wins" if row["mean_ratio"] > 1 else "Ablation wins"
            print(f"    budget={row['prop_budget']:.3f}  |  ratio={row['mean_ratio']:.4f}  |  {marker}  |  n={int(row['n_seeds'])}")

plt.suptitle("Figure 5: CI-Width Ratio by Missingness Setting (2.5%-50%)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/figure5_by_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: figures/figure5_by_missingness.png")
