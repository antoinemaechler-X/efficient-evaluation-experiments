# WOR-FAQ Study — Reproduction Package

This folder reproduces the Without-Replacement (WOR) FAQ experiments from
**"Efficiently Evaluating LLM Performance with Statistical Guarantees"**.

---

## What This Studies

We extend FAQ to sample questions **without replacement** using equation (2) from the paper:

> θ̂_n = (1/n) Σ_t φ_t,  where φ_t uses actual y_i for already-observed questions

The key change is the WOR PAI estimator: the imputed sum uses real labels for observed
questions and model predictions for unobserved ones. The variance formula also changes
(Theorem 4 of the PAI paper).

**Key finding**: WOR-FAQ substantially outperforms the original with-replacement best
baseline on both datasets (MMLU-Pro and BBH+GPQA+IFEval+MATH+MuSR).

---

## Correspondence to Original Repo

Each file here is a direct WOR replacement of a file from [skbwu/efficiently-evaluating-llms](https://github.com/skbwu/efficiently-evaluating-llms):

| This file | Replaces (original repo) | What changed |
|-----------|--------------------------|--------------|
| `wor_trial.py` | *(new — analogous to inline trial code in `faq_final.py`)* | WOR sampling + WOR variance formula |
| `wor_faq_val.py` | `faq_val.py` | Uses `trial_faq_wor`, only fully-observed data |
| `wor_val_analyzer.py` | `faq_val_analyzer.py` | Reads `wor_faq_val_*.csv` instead |
| `wor_faq_final.py` | `faq_final.py` | Uses `trial_faq_wor`, only fully-observed data |
| `wor_baselines.py` | `baselines_all.py` | WOR sampling, WOR variance, static PHATS predictor |
| `wor_ablation.py` | `active_inference_factor_ablation.py` | Z&C scoring adapted for WOR |

---

## Quick Start

### 1. Explore results in the notebook

Open `analysis.ipynb` to:
- Load and visualize all precomputed results (in `results/`)
- Compare WOR-FAQ against original with-replacement baselines
- Examine the ablation (FAQ scoring vs. simpler Z&C scoring, both WOR)

The notebook **does not require running any experiments** — all results are precomputed in `results/`.

### 2. Get the data (only needed to re-run cluster scripts)

The preprocessed data (`data/processed/`) and factor models (`factor_models/`) are required
to re-run the cluster jobs. See the original repo README for download instructions.

### 3. Reproduce experiments on a cluster (optional)

Typical command sequence:

```bash
# 1. Validation sweep (tune hyperparameters)
sbatch submit_wor_faq_val.sh

# 2. Find best hyperparameters
python wor_val_analyzer.py  # outputs logs/val/wor_best_settings.csv

# 3. Final runs
sbatch submit_wor_faq_final.sh

# 4. Baselines and ablation (curiosity)
sbatch submit_wor_baselines.sh
sbatch submit_wor_ablation.sh
```

---

## File Structure

```
wor_study/
├── README.md

├── wor_trial.py          # WOR trial functions: trial_faq_wor, trial_ablation_wor
│
├── wor_faq_val.py        # hyperparameter validation (replaces faq_val.py)
├── wor_val_analyzer.py   # pick best settings from val CSVs (replaces faq_val_analyzer.py)
├── wor_faq_final.py      # final test runs (replaces faq_final.py)
├── wor_baselines.py      # static-predictor baselines (replaces baselines_all.py)
├── wor_ablation.py       # Z&C scoring ablation (replaces active_inference_factor_ablation.py)
│
├── submit_wor_faq_val.sh      # SLURM: 10 jobs (2 datasets × 5 gammas)
├── submit_wor_faq_final.sh    # SLURM: 3 jobs (seed chunks)
├── submit_wor_baselines.sh    # SLURM: 6 jobs (2 datasets × 3 seed chunks)
├── submit_wor_ablation.sh     # SLURM: 6 jobs (2 datasets × 3 seed chunks)
│
├── analysis.ipynb        # load results, compute ESS, plot figures
│
└── results/
    ├── wor_faq_final_sl=*.csv                    # WOR FAQ per-seed results (3 files)
    ├── wor_baselines_dataset=*_sl=*.csv          # WOR baselines (6 files)
    ├── wor_ablation_dataset=*_sl=*.csv           # WOR ablation (6 files)
    ├── wor_vs_orig_best_baseline_summary.csv     # original WR best baseline summary
    ├── wor_vs_orig_uniform_summary.csv           # original WR uniform summary
    ├── wor_ess+coverage_fully-observed.pdf       # main figure
    └── wor_ess+coverage_ablation.pdf             # ablation figure
```

---

## Dependencies

Same environment as the main repo:

```bash
conda activate faq_env
```

Required packages: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`.
