'''
Task 2: Extract post-hoc best tau for the traditional active inference ablation
at 25%-50% budget values, for ALL missingness settings.

For each (dataset, missingness, budget), picks the tau with smallest mean_width
averaged across 100 seeds.

Saves results to:
  logs/final/ablation_posthoc_best.csv (all missingness settings)
  logs/final/ablation_posthoc_all.csv

Usage: python analyze_ablation_posthoc.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

os.makedirs("figures", exist_ok=True)

DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]

# load and concatenate both datasets
dfs = []
for dataset in DATASETS:
    df = pd.read_csv(f"logs/final/active_inference_ablation_dataset={dataset}.csv")
    dfs.append(df)
    print(f"Loaded {dataset}: {len(df)} rows")
ablation_df = pd.concat(dfs, ignore_index=True)

# filter to 25%-50% budgets only
ablation_df = ablation_df[ablation_df["prop_budget"] > 0.25].copy()
ablation_df["prop_budget"] = np.round(ablation_df["prop_budget"], decimals=3)

print(f"\nTotal rows (>25% budgets): {len(ablation_df)}")
print(f"Unique budgets: {sorted(ablation_df['prop_budget'].unique())}")
print(f"Unique taus: {sorted(ablation_df['tau'].unique())}")
print(f"Unique missingness settings:")
for (nfo, mcp), g in ablation_df.groupby(["n_full_obs", "mcar_obs_prob"]):
    print(f"  n_full_obs={nfo}, mcar_obs_prob={mcp}: {len(g)} rows")

# average across 100 seeds per (dataset, n_full_obs, mcar_obs_prob, budget, tau)
variant_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau"]
means = ablation_df.groupby(variant_cols, dropna=False).agg(
    mean_width=("mean_width", "mean"),
    mean_width_se=("mean_width", lambda x: x.std() / np.sqrt(len(x))),
    coverage=("coverage", "mean"),
    n_seeds=("seed", "count")
).reset_index()

# pick best tau (smallest mean_width) per (dataset, missingness, budget)
scenario_cols = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"]
best_idx = means.groupby(scenario_cols, dropna=False)["mean_width"].idxmin()
best_settings = means.loc[best_idx].sort_values(["dataset", "mcar_obs_prob", "prop_budget"]).reset_index(drop=True)

# display results
print("\n" + "=" * 90)
print("POST-HOC BEST TAU PER (DATASET, MISSINGNESS, BUDGET)")
print("=" * 90)
for dataset in DATASETS:
    sub = best_settings[best_settings["dataset"] == dataset]
    for (nfo, mcp), grp in sub.groupby(["n_full_obs", "mcar_obs_prob"]):
        print(f"\n--- {dataset} | n_full_obs={nfo}, mcar_obs_prob={mcp} ---")
        for _, row in grp.iterrows():
            print(f"  budget={row['prop_budget']:.3f}  |  best tau={row['tau']:.2f}  |  "
                  f"mean_width={row['mean_width']:.6f}  |  coverage={row['coverage']:.4f}")

# save
best_settings.to_csv("logs/final/ablation_posthoc_best.csv", index=False)
means.to_csv("logs/final/ablation_posthoc_all.csv", index=False)
print(f"\nSaved: logs/final/ablation_posthoc_best.csv ({len(best_settings)} rows)")
print(f"Saved: logs/final/ablation_posthoc_all.csv ({len(means)} rows)")

# ============================================================
# PLOT 1: Mean CI width by tau — fully-observed (one subplot per dataset)
# ============================================================
tau_colors = {0.05: "firebrick", 0.25: "darkorange", 0.5: "steelblue", 0.75: "seagreen"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, dataset in enumerate(DATASETS):
    ax = axes[i]
    subset = means[(means["dataset"] == dataset) & (means["mcar_obs_prob"] == 1.0)]
    for tau in sorted(subset["tau"].unique()):
        t = subset[subset["tau"] == tau].sort_values("prop_budget")
        ax.plot(t["prop_budget"], t["mean_width"], "o-", color=tau_colors.get(tau, "gray"),
                label=f"tau={tau:.2f}", markersize=5)
    best_sub = best_settings[(best_settings["dataset"] == dataset) & (best_settings["mcar_obs_prob"] == 1.0)]
    ax.plot(best_sub["prop_budget"], best_sub["mean_width"], "k*", markersize=12,
            label="Post-hoc best", zorder=5)
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("Mean CI Width", fontsize=12)
    ax.set_title(f"{dataset}\n(fully-observed)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Active Inference Ablation: CI Width by Tau (25%-50%, fully-obs)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/task2_ablation_width_by_tau.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: figures/task2_ablation_width_by_tau.png")

# ============================================================
# PLOT 2: Best tau bar chart — fully-observed
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, dataset in enumerate(DATASETS):
    ax = axes[i]
    best_sub = best_settings[(best_settings["dataset"] == dataset) & (best_settings["mcar_obs_prob"] == 1.0)].sort_values("prop_budget")
    colors = [tau_colors.get(t, "gray") for t in best_sub["tau"]]
    ax.bar(range(len(best_sub)), best_sub["mean_width"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(best_sub)))
    ax.set_xticklabels([f"{b:.3f}" for b in best_sub["prop_budget"]], rotation=45, fontsize=9)
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("Best Mean CI Width", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=f"tau={t:.2f}")
                       for t, c in tau_colors.items()]
    ax.legend(handles=legend_elements, fontsize=9, title="Best tau")
    ax.grid(alpha=0.3, axis="y")

plt.suptitle("Post-hoc Best Tau per Budget (fully-obs, 25%-50%)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/task2_ablation_best_tau.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/task2_ablation_best_tau.png")

# ============================================================
# PLOT 3: Comparison across missingness settings
# ============================================================
missingness_labels = {
    1.0: "Fully observed",
    0.01: "MCAR p=0.01",
    0.001: "MCAR p=0.001",
    0.0001: "MCAR p=0.0001",
    0.00001: "MCAR p=0.00001",
}
ms_colors = {1.0: "darkblue", 0.01: "steelblue", 0.001: "darkorange", 0.0001: "firebrick", 0.00001: "seagreen"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, dataset in enumerate(DATASETS):
    ax = axes[i]
    for mcp in sorted(best_settings["mcar_obs_prob"].unique()):
        sub = best_settings[(best_settings["dataset"] == dataset) & (best_settings["mcar_obs_prob"] == mcp)].sort_values("prop_budget")
        if len(sub) == 0:
            continue
        ax.plot(sub["prop_budget"], sub["mean_width"], "o-",
                color=ms_colors.get(mcp, "gray"), markersize=5,
                label=missingness_labels.get(mcp, f"p={mcp}"))
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("Best Mean CI Width (post-hoc)", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("Ablation: Best CI Width by Missingness Setting (25%-50%)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/task2_ablation_by_missingness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/task2_ablation_by_missingness.png")
