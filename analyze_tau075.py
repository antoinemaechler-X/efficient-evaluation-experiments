'''
Analyze FAQ with tau=0.75 vs traditional active inference (post-hoc best).
Computes CI-width ratio per seed and plots comparison.

Both use 2.5%-25% budgets on fully-observed data.

Requires:
  - logs/tau075/final_{dataset}.csv (tau=0.75 FAQ results, 100 seeds)
  - logs/final/active_inference_ablation_logs.csv (original paper ablation, 2.5%-25%)

Usage: python analyze_tau075.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures/tau075", exist_ok=True)

DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]

# ---- Load tau=0.75 FAQ final results ----
tau075_dfs = []
for dataset in DATASETS:
    fname = f"logs/tau075/final_{dataset}.csv"
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        tau075_dfs.append(df)
        print(f"Loaded {fname}: {len(df)} rows")
    else:
        print(f"MISSING: {fname}")

tau075_df = pd.concat(tau075_dfs, ignore_index=True)

# ---- Load original ablation results (2.5%-25% budgets, from paper) ----
ablation_all = pd.read_csv("logs/final/active_inference_ablation_logs.csv")
print(f"Loaded ablation_logs: {len(ablation_all)} rows")

# filter to fully-observed only
ablation_all = ablation_all[ablation_all["mcar_obs_prob"] == 1.0]

# round budgets to 3 decimals for matching
ablation_all["prop_budget"] = np.round(ablation_all["prop_budget"], decimals=3)

print(f"Fully-observed ablation rows: {len(ablation_all)}")
print(f"Ablation budgets: {sorted(ablation_all['prop_budget'].unique())}")
print(f"Tau075 budgets: {sorted(tau075_df['prop_budget'].unique())}")

# pick post-hoc best tau per (dataset, budget)
abl_means = ablation_all.groupby(["dataset", "prop_budget", "tau"]).agg(
    mean_width=("mean_width", "mean")
).reset_index()
best_idx = abl_means.groupby(["dataset", "prop_budget"])["mean_width"].idxmin()
ablation_best = abl_means.loc[best_idx][["dataset", "prop_budget", "tau"]]
print(f"\nPost-hoc best tau per budget:")
for _, row in ablation_best.iterrows():
    print(f"  {row['dataset']}: budget={row['prop_budget']:.3f}, best tau={row['tau']:.2f}")

# get per-seed results for the post-hoc best tau
best_ablation_per_seed = pd.merge(
    ablation_all,
    ablation_best[["dataset", "prop_budget", "tau"]],
    on=["dataset", "prop_budget", "tau"],
    how="inner"
)
print(f"Ablation per-seed rows (post-hoc best): {len(best_ablation_per_seed)}")

# ============================================================
# PLOT 1: CI-width ratio (Ablation / tau=0.75 FAQ) per budget
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, dataset in enumerate(DATASETS):
    ax = axes[i]

    faq_sub = tau075_df[tau075_df["dataset"] == dataset]
    abl_sub = best_ablation_per_seed[best_ablation_per_seed["dataset"] == dataset]

    merged = pd.merge(
        faq_sub[["prop_budget", "seed", "mean_width"]].rename(columns={"mean_width": "faq_width"}),
        abl_sub[["prop_budget", "seed", "mean_width"]].rename(columns={"mean_width": "abl_width"}),
        on=["prop_budget", "seed"],
        how="inner"
    )

    if len(merged) == 0:
        print(f"WARNING: no merged data for {dataset}")
        ax.set_title(f"{dataset}\nNO DATA")
        continue

    merged["ratio"] = merged["abl_width"] / merged["faq_width"]

    summary = merged.groupby("prop_budget").agg(
        mean_ratio=("ratio", "mean"),
        se_ratio=("ratio", lambda x: x.std() / np.sqrt(len(x))),
        n_seeds=("ratio", "count")
    ).reset_index()

    ax.errorbar(summary["prop_budget"], summary["mean_ratio"],
                yerr=1.96 * summary["se_ratio"], fmt="o-", capsize=3,
                color="darkblue", label="Ablation / FAQ(tau=0.75)")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Ratio = 1 (equal)")
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("CI-Width Ratio", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    print(f"\n--- {dataset}: Ablation / FAQ(tau=0.75) ---")
    for _, row in summary.iterrows():
        winner = "FAQ wins" if row["mean_ratio"] > 1 else "Ablation wins"
        print(f"  budget={row['prop_budget']:.3f}  |  ratio={row['mean_ratio']:.4f} ± {row['se_ratio']:.4f}  |  {winner}  |  n={int(row['n_seeds'])}")

plt.suptitle("CI-Width Ratio: Post-hoc Best Ablation / FAQ(tau=0.75)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/tau075/ratio_ablation_vs_tau075.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: figures/tau075/ratio_ablation_vs_tau075.png")

# ============================================================
# PLOT 2: Direct CI width comparison (3 lines: FAQ tau=0.75, ablation, ablation mean)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, dataset in enumerate(DATASETS):
    ax = axes[i]

    # tau=0.75 FAQ: average across seeds
    faq_sub = tau075_df[tau075_df["dataset"] == dataset]
    faq_summary = faq_sub.groupby("prop_budget").agg(
        mean_width=("mean_width", "mean"),
        se=("mean_width", lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()

    # ablation post-hoc best: average across seeds
    abl_sub = best_ablation_per_seed[best_ablation_per_seed["dataset"] == dataset]
    abl_summary = abl_sub.groupby("prop_budget").agg(
        mean_width=("mean_width", "mean"),
        se=("mean_width", lambda x: x.std() / np.sqrt(len(x)))
    ).reset_index()

    ax.errorbar(faq_summary["prop_budget"], faq_summary["mean_width"],
                yerr=1.96 * faq_summary["se"], fmt="o-", capsize=3,
                color="steelblue", label="FAQ (tau=0.75)")
    ax.errorbar(abl_summary["prop_budget"], abl_summary["mean_width"],
                yerr=1.96 * abl_summary["se"], fmt="s-", capsize=3,
                color="firebrick", label="Ablation (post-hoc best)")

    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("Mean CI Width", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Mean CI Width: FAQ(tau=0.75) vs Post-hoc Best Ablation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/tau075/width_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/tau075/width_comparison.png")

# ============================================================
# PLOT 3: Coverage comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, dataset in enumerate(DATASETS):
    ax = axes[i]

    faq_sub = tau075_df[tau075_df["dataset"] == dataset]
    faq_cov = faq_sub.groupby("prop_budget")["coverage"].mean().reset_index()

    abl_sub = best_ablation_per_seed[best_ablation_per_seed["dataset"] == dataset]
    abl_cov = abl_sub.groupby("prop_budget")["coverage"].mean().reset_index()

    ax.plot(faq_cov["prop_budget"], faq_cov["coverage"], "o-", color="steelblue", label="FAQ (tau=0.75)")
    ax.plot(abl_cov["prop_budget"], abl_cov["coverage"], "s-", color="firebrick", label="Ablation (post-hoc best)")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.7, label="95% target")
    ax.set_xlabel("Budget Proportion", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(dataset, fontsize=13)
    ax.set_ylim(0.93, 0.96)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle("Coverage: FAQ(tau=0.75) vs Post-hoc Best Ablation", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("figures/tau075/coverage_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: figures/tau075/coverage_comparison.png")

# ============================================================
# Summary table
# ============================================================
print(f"\n{'='*90}")
print("SUMMARY TABLE")
print(f"{'='*90}")
for dataset in DATASETS:
    print(f"\n--- {dataset} ---")
    print(f"  {'budget':>7}  {'FAQ(0.75) width':>16}  {'Ablation width':>16}  {'Ratio':>7}  {'FAQ cov':>8}  {'Abl cov':>8}")
    print(f"  {'-'*75}")

    faq_sub = tau075_df[tau075_df["dataset"] == dataset]
    abl_sub = best_ablation_per_seed[best_ablation_per_seed["dataset"] == dataset]

    for budget in sorted(faq_sub["prop_budget"].unique()):
        faq_b = faq_sub[faq_sub["prop_budget"] == budget]
        abl_b = abl_sub[abl_sub["prop_budget"] == budget]

        if len(faq_b) == 0 or len(abl_b) == 0:
            continue

        faq_w = faq_b["mean_width"].mean()
        abl_w = abl_b["mean_width"].mean()
        ratio = abl_w / faq_w
        faq_c = faq_b["coverage"].mean()
        abl_c = abl_b["coverage"].mean()

        marker = "  <-- FAQ wins" if ratio > 1 else ""
        print(f"  {budget:>7.3f}  {faq_w:>16.6f}  {abl_w:>16.6f}  {ratio:>7.4f}  {faq_c:>8.4f}  {abl_c:>8.4f}{marker}")

print("\nDone!")
