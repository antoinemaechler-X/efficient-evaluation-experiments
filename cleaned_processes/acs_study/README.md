# ACS Census Study — Reproduction Package

This folder reproduces the ACS Census 2019 (California) experiments from
**"Efficiently Evaluating LLM Performance with Statistical Guarantees"**.

---

## What This Studies

We test the FAQ algorithm on a **continuous regression** setting:
- **Dataset**: ACS PUMS 2019, California, ~195k individuals
- **Estimand**: Mean of normalized personal income (PINCP) over an unlabeled split
- **Estimators**: classical (no ML), uniform+PAI (AIPW with uniform sampling), FAQ (AIPW with active sampling)

**Key finding**: On this continuous outcome, FAQ converges to uniform+PAI — active sampling provides no additional benefit. The root cause is that ACS income residuals are homoscedastic (FAQ needs heteroscedastic uncertainty to prioritize sampling). However, AIPW-based methods (uniform+PAI) still beat classical by ~1.3× (BLR) to ~1.6× (XGBoost).

---

## Quick Start

### 1. Get the data

The ACS data file (`psam_p06.csv`, ~257 MB) is needed to run the experiments.
It is **not included** due to size — download it via:

```bash
pip install folktables
python -c "
import folktables
src = folktables.ACSDataSource(survey_year=2019, horizon='1-Year', survey='person', root_dir='data')
src.get_data(states=['CA'], download=True)
"
```

This saves the file to `data/2019/1-Year/psam_p06.csv`.

### 2. Explore results in the notebook

Open `analysis.ipynb` to:
- Load and visualize all precomputed results (in `results/`)
- Understand why FAQ collapses to uniform+PAI (posterior diagnostic)
- Compare all methods and experiments

The notebook **does not require running any experiments** — all results are precomputed in `results/`.

### 3. Reproduce experiments on a cluster (optional)

See **Cluster Scripts** below.

---

## File Structure

```
acs_study/
├── README.md

├── utils.py             # data loading and feature encoding
│
├── train_model.py       # train XGBoost income predictor (needed for run_baseline.py)
├── run_baseline.py      # classical / uniform / active inference on OLS coefficient
├── run_faq.py           # FAQ with BLR predictions (main experiment)
├── run_faq_xgb.py       # FAQ with XGBoost + BLR predictions (hybrid)
│
├── plot_paper.py        # paper-style ESS + coverage figures from CSV
│
├── submit_faq_acs.sh          # SLURM: D ∈ {8, 16, 32} FAQ runs
├── submit_faq_acs_lowlabel.sh # SLURM: n_labeled ∈ {100, 500, 2000} ablation
├── submit_faq_acs_nlab50.sh   # SLURM: n_labeled = 50
├── submit_faq_acs_xgb.sh      # SLURM: XGBoost+BLR hybrid experiment
│
├── analysis.ipynb       # comprehensive analysis notebook
│
└── results/
    ├── *.csv            # precomputed results from cluster runs
    └── *.png            # precomputed figures
```

---

## Cluster Scripts

Each SLURM script submits an array job. Adapt the header (account, partition, module paths) to your cluster.

| Script | Experiment | Output CSVs |
|--------|-----------|-------------|
| `submit_faq_acs.sh` | D=8,16,32 BLR-FAQ | `faq_acs_D{8,16,32}.csv` |
| `submit_faq_acs_lowlabel.sh` | n_labeled ∈ {100, 500, 2000} | `faq_acs_nlab{100,500,2000}.csv` |
| `submit_faq_acs_nlab50.sh` | n_labeled = 50 | `faq_acs_nlab50.csv` |
| `submit_faq_acs_xgb.sh` | XGBoost+BLR hybrid | `faq_acs_xgb_*.csv` |

**Typical command sequence (local, one experiment):**
```bash
# 1. Run FAQ experiment (BLR, D=16)
python run_faq.py --D 16 --num_trials 100 --out_csv faq_acs_D16.csv

# 2. Plot
python plot_paper.py faq_acs_D16.csv --out faq_acs_D16_figure2.png
```

For full-scale runs (100 trials, 11 budgets, full ~95k unlabeled points), use the SLURM scripts on a cluster.

---

## Dependencies

Reproduce the exact environment with:
```bash
conda env create -f environment.yml
conda activate acs_env
```
