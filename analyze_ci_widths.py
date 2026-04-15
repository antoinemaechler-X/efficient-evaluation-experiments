'''
Task 4: Analyze distribution of CI widths across ~2.2K test models.
Loads per-model width vectors from .npy files, averages across 100 seeds,
and produces distribution statistics + plots.

Includes outlier identification with model metadata analysis
(org type, model size, release date).

Usage: python analyze_ci_widths.py
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import re

DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
N_SEEDS = 100

MAJOR_LABS = {
    "meta-llama", "google", "microsoft", "CohereForAI", "nvidia",
    "Qwen", "mistralai", "deepseek-ai",
}

os.makedirs("figures/ci_widths", exist_ok=True)


def extract_org(model_name):
    """Extract org from 'org__model' format."""
    if "__" in model_name:
        return model_name.split("__")[0]
    return "unknown"


def classify_org(org):
    """Classify org as 'Major Lab' or 'Community'."""
    return "Major Lab" if org in MAJOR_LABS else "Community"


def extract_model_size(model_name):
    """Extract parameter count in billions from model name (e.g., '8B' -> 8.0)."""
    match = re.search(r'(\d+(?:\.\d+)?)[Bb](?!\w)', model_name)
    if match:
        return float(match.group(1))
    return None


def load_mean_widths(dataset, budget):
    """Load and average widths across seeds for a (dataset, budget) pair."""
    widths_all = []
    for seed in range(N_SEEDS):
        fname = f"logs/ci_widths/{dataset}_budget={budget}_seed={seed}.npy"
        if os.path.exists(fname):
            widths_all.append(np.load(fname))
    if len(widths_all) == 0:
        return None
    return np.stack(widths_all).mean(axis=0)


for dataset in DATASETS:
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset}")
    print(f"{'='*80}")

    for budget in BUDGET_PROPS:
        mean_widths = load_mean_widths(dataset, budget)
        if mean_widths is None:
            print(f"  budget={budget:.3f}: NO DATA")
            continue

        n_models = len(mean_widths)
        print(f"\n  budget={budget:.3f} ({n_models} models):")
        print(f"    Overall mean:   {mean_widths.mean():.6f}")
        print(f"    Median:         {np.median(mean_widths):.6f}")
        print(f"    Std:            {mean_widths.std():.6f}")
        print(f"    Min:            {mean_widths.min():.6f}")
        print(f"    Max:            {mean_widths.max():.6f}")
        print(f"    5th percentile: {np.percentile(mean_widths, 5):.6f}")
        print(f"    25th pctile:    {np.percentile(mean_widths, 25):.6f}")
        print(f"    75th pctile:    {np.percentile(mean_widths, 75):.6f}")
        print(f"    95th pctile:    {np.percentile(mean_widths, 95):.6f}")
        print(f"    Max/Min ratio:  {mean_widths.max() / mean_widths.min():.1f}x")

    # ---- PLOT 1: Histograms at select budgets ----
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f"CI Width Distribution Across Models — {dataset}", fontsize=14)

    for i, budget in enumerate(BUDGET_PROPS):
        ax = axes[i // 5, i % 5]
        mean_widths = load_mean_widths(dataset, budget)

        if mean_widths is None:
            ax.set_title(f"budget={budget:.3f}\nNO DATA")
            continue

        ax.hist(mean_widths, bins=50, edgecolor="black", linewidth=0.5, alpha=0.7)
        ax.axvline(mean_widths.mean(), color="red", linestyle="--", label=f"mean={mean_widths.mean():.4f}")
        ax.axvline(np.median(mean_widths), color="blue", linestyle="--", label=f"med={np.median(mean_widths):.4f}")
        ax.set_title(f"budget={budget:.3f}")
        ax.legend(fontsize=7)
        ax.set_xlabel("CI Width")

    plt.tight_layout()
    plt.savefig(f"figures/ci_widths/histograms_{dataset}.png", dpi=150)
    plt.close()
    print(f"\n  Saved: figures/ci_widths/histograms_{dataset}.png")

    # ---- Load model metadata ----
    M2_meta = pd.read_csv(f"data/processed/{dataset}/M2.csv", usecols=["model", "created_date"])
    M2_meta["org"] = M2_meta["model"].apply(extract_org)
    M2_meta["org_type"] = M2_meta["org"].apply(classify_org)
    M2_meta["model_size_B"] = M2_meta["model"].apply(extract_model_size)
    M2_meta["created_date"] = pd.to_datetime(M2_meta["created_date"], errors="coerce")
    n_models = len(M2_meta)

    # ---- PLOT 2: Boxplots across budgets + flag outliers ----
    fig, ax = plt.subplots(figsize=(12, 6))
    boxplot_data = []
    labels = []
    outlier_records = []

    for budget in BUDGET_PROPS:
        mean_widths = load_mean_widths(dataset, budget)
        if mean_widths is None:
            continue

        boxplot_data.append(mean_widths)
        labels.append(f"{budget:.3f}")

        # flag outliers (same rule as matplotlib boxplot: outside 1.5*IQR)
        q1, q3 = np.percentile(mean_widths, 25), np.percentile(mean_widths, 75)
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        high_fence = q3 + 1.5 * iqr
        outlier_mask = (mean_widths < low_fence) | (mean_widths > high_fence)
        outlier_idx = np.where(outlier_mask)[0]

        for idx in outlier_idx:
            row = M2_meta.iloc[idx] if idx < len(M2_meta) else None
            model_name = row["model"] if row is not None else f"model_{idx}"
            outlier_records.append({
                "dataset": dataset,
                "budget": budget,
                "model_idx": idx,
                "model_name": model_name,
                "org": extract_org(model_name),
                "org_type": classify_org(extract_org(model_name)),
                "model_size_B": extract_model_size(model_name),
                "created_date": row["created_date"] if row is not None else None,
                "ci_width": mean_widths[idx],
                "median_width": np.median(mean_widths),
                "ratio_to_median": mean_widths[idx] / np.median(mean_widths),
                "direction": "high" if mean_widths[idx] > high_fence else "low"
            })

    bp = ax.boxplot(boxplot_data, labels=labels, showfliers=True,
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4, alpha=0.5))
    ax.set_xlabel("Budget Proportion")
    ax.set_ylabel("CI Width (averaged across seeds)")
    ax.set_title(f"CI Width Distribution Across ~{n_models} Test Models — {dataset}")
    plt.tight_layout()
    plt.savefig(f"figures/ci_widths/boxplots_{dataset}.png", dpi=150)
    plt.close()
    print(f"  Saved: figures/ci_widths/boxplots_{dataset}.png")

    # save outlier report
    if outlier_records:
        outlier_df = pd.DataFrame(outlier_records)
        outlier_df.to_csv(f"figures/ci_widths/outliers_{dataset}.csv", index=False)
        print(f"  Saved: figures/ci_widths/outliers_{dataset}.csv")

        # print summary
        print(f"\n  OUTLIER SUMMARY:")
        freq = outlier_df.groupby(["model_name", "direction"]).size().reset_index(name="n_budgets")
        freq = freq.sort_values("n_budgets", ascending=False)
        print(f"  Models that are outliers across the most budgets:")
        for _, row in freq.head(10).iterrows():
            print(f"    {row['model_name']}: outlier in {row['n_budgets']}/10 budgets ({row['direction']})")

    # ---- PLOT 3: 95th/5th percentile ratio across budgets ----
    ratios = []
    budgets_used = []
    for budget in BUDGET_PROPS:
        mean_widths = load_mean_widths(dataset, budget)
        if mean_widths is None:
            continue
        ratios.append(np.percentile(mean_widths, 95) / np.percentile(mean_widths, 5))
        budgets_used.append(budget)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(budgets_used, ratios, "o-", color="darkblue")
    ax.set_xlabel("Budget Proportion")
    ax.set_ylabel("95th / 5th Percentile Ratio")
    ax.set_title(f"Spread of CI Widths Across Models — {dataset}")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"figures/ci_widths/spread_ratio_{dataset}.png", dpi=150)
    plt.close()
    print(f"  Saved: figures/ci_widths/spread_ratio_{dataset}.png")

    # ==================================================================
    # PLOTS 4a-4e: Model metadata analysis (runs for all datasets)
    # ==================================================================

    # Use a representative budget (median budget) for per-model analysis
    ref_budget = BUDGET_PROPS[len(BUDGET_PROPS) // 2]
    ref_widths = load_mean_widths(dataset, ref_budget)
    if ref_widths is None:
        ref_budget = BUDGET_PROPS[0]
        ref_widths = load_mean_widths(dataset, ref_budget)

    # Build per-model dataframe at reference budget
    model_df = M2_meta.copy()
    model_df["ci_width"] = ref_widths[:len(model_df)]

    # Identify outliers at this budget
    q1, q3 = np.percentile(ref_widths, 25), np.percentile(ref_widths, 75)
    iqr = q3 - q1
    model_df["is_outlier"] = (model_df["ci_width"] < q1 - 1.5 * iqr) | (model_df["ci_width"] > q3 + 1.5 * iqr)

    # --- Plot 4a: CI width vs model size, colored by org type ---
    has_size = model_df.dropna(subset=["model_size_B"])
    if len(has_size) > 50:
        fig, ax = plt.subplots(figsize=(10, 6))

        for org_type, color, marker in [("Community", "#7fb3d8", "o"), ("Major Lab", "#e74c3c", "D")]:
            sub = has_size[has_size["org_type"] == org_type]
            ax.scatter(sub["model_size_B"], sub["ci_width"],
                       c=color, marker=marker, alpha=0.3, s=20, label=f"{org_type} ({len(sub)})")

        # highlight outliers
        outlier_sub = has_size[has_size["is_outlier"]]
        if len(outlier_sub) > 0:
            ax.scatter(outlier_sub["model_size_B"], outlier_sub["ci_width"],
                       facecolors="none", edgecolors="black", s=80, linewidths=1.5,
                       label=f"Outliers ({len(outlier_sub)})", zorder=5)

        ax.set_xlabel("Model Size (B parameters)", fontsize=12)
        ax.set_ylabel(f"CI Width (budget={ref_budget:.3f})", fontsize=12)
        ax.set_title(f"CI Width vs Model Size — {dataset}", fontsize=13)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"figures/ci_widths/size_vs_width_{dataset}.png", dpi=150)
        plt.close()
        print(f"  Saved: figures/ci_widths/size_vs_width_{dataset}.png")

    # --- Plot 4b: Boxplot of CI width by model size bucket ---
    if len(has_size) > 50:
        size_bins = [0, 1, 3, 7, 13, 35, 100]
        size_labels = ["<1B", "1-3B", "3-7B", "7-13B", "13-35B", "35B+"]
        has_size = has_size.copy()
        has_size["size_bucket"] = pd.cut(has_size["model_size_B"], bins=size_bins, labels=size_labels, right=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        groups = [g["ci_width"].values for _, g in has_size.groupby("size_bucket", observed=True)]
        group_labels = [f"{l}\n(n={len(g)})" for l, (_, g) in
                        zip(size_labels, has_size.groupby("size_bucket", observed=True))]

        if groups:
            bp = ax.boxplot(groups, labels=group_labels, showfliers=True,
                            flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.4))
            ax.set_xlabel("Model Size", fontsize=12)
            ax.set_ylabel(f"CI Width (budget={ref_budget:.3f})", fontsize=12)
            ax.set_title(f"CI Width by Model Size — {dataset}", fontsize=13)
            ax.grid(alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(f"figures/ci_widths/size_boxplot_{dataset}.png", dpi=150)
            plt.close()
            print(f"  Saved: figures/ci_widths/size_boxplot_{dataset}.png")

    # --- Plot 4c: Org type comparison (Major Lab vs Community) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    community = model_df[model_df["org_type"] == "Community"]["ci_width"].dropna()
    major_lab = model_df[model_df["org_type"] == "Major Lab"]["ci_width"].dropna()

    data_to_plot = []
    bp_labels = []
    if len(community) > 0:
        data_to_plot.append(community.values)
        bp_labels.append(f"Community\n(n={len(community)})")
    if len(major_lab) > 0:
        data_to_plot.append(major_lab.values)
        bp_labels.append(f"Major Lab\n(n={len(major_lab)})")

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=bp_labels, showfliers=True,
                        flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.4),
                        patch_artist=True)
        colors = ["#7fb3d8", "#e74c3c"][:len(data_to_plot)]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.set_ylabel(f"CI Width (budget={ref_budget:.3f})", fontsize=12)
        ax.set_title(f"CI Width: Major Labs vs Community — {dataset}", fontsize=13)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(f"figures/ci_widths/org_type_{dataset}.png", dpi=150)
        plt.close()
        print(f"  Saved: figures/ci_widths/org_type_{dataset}.png")

    # --- Plot 4d: CI width over time (release date) ---
    has_date = model_df.dropna(subset=["created_date"])
    if len(has_date) > 50:
        fig, ax = plt.subplots(figsize=(12, 5))

        non_outlier = has_date[~has_date["is_outlier"]]
        outlier = has_date[has_date["is_outlier"]]

        ax.scatter(non_outlier["created_date"], non_outlier["ci_width"],
                   c="#7fb3d8", alpha=0.2, s=10, label=f"Normal ({len(non_outlier)})")
        if len(outlier) > 0:
            ax.scatter(outlier["created_date"], outlier["ci_width"],
                       c="#e74c3c", alpha=0.7, s=25, label=f"Outliers ({len(outlier)})", zorder=5)

        # rolling average: sort by date, use centered 7-day resample
        has_date_sorted = has_date.sort_values("created_date").set_index("created_date")
        weekly = has_date_sorted["ci_width"].resample("7D").mean().dropna()
        ax.plot(weekly.index, weekly.values, color="black", linewidth=2, alpha=0.7, label="Weekly avg")

        ax.set_xlabel("Model Release Date", fontsize=12)
        ax.set_ylabel(f"CI Width (budget={ref_budget:.3f})", fontsize=12)
        ax.set_title(f"CI Width Over Time — {dataset}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f"figures/ci_widths/date_vs_width_{dataset}.png", dpi=150)
        plt.close()
        print(f"  Saved: figures/ci_widths/date_vs_width_{dataset}.png")

    # --- Plot 4e: Top outlier orgs bar chart (only if outliers exist) ---
    if not outlier_records:
        print(f"\n  No outliers detected for {dataset}, skipping outlier org chart.")
    else:
        outlier_df = pd.DataFrame(outlier_records)
        unique_outliers = outlier_df.drop_duplicates(subset=["model_name"])
        org_counts = unique_outliers["org"].value_counts().head(15)

    if outlier_records and len(org_counts) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = ["#e74c3c" if org in MAJOR_LABS else "#7fb3d8" for org in org_counts.index]
        ax.barh(range(len(org_counts)), org_counts.values, color=colors_bar, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(org_counts)))
        ax.set_yticklabels(org_counts.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Number of Outlier Models", fontsize=12)
        ax.set_title(f"Top Orgs with Outlier CI Widths — {dataset}", fontsize=13)
        ax.legend(handles=[Patch(facecolor="#e74c3c", label="Major Lab"),
                           Patch(facecolor="#7fb3d8", label="Community")],
                  fontsize=9)
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(f"figures/ci_widths/outlier_orgs_{dataset}.png", dpi=150)
        plt.close()
        print(f"  Saved: figures/ci_widths/outlier_orgs_{dataset}.png")

print("\nDone!")
