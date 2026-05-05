# AlphaFold Study — Reproduction Package

Applies WOR-PAI active inference to the AlphaFold proteomics dataset,
comparing against Bernoulli-based active inference (Zrnic & Candès) and classical estimation.

---

## What This Studies

We estimate mean disorder probability separately for two groups:
- **Non-phosphorylated** proteins
- **Phosphorylated** proteins

AlphaFold predictions Ŷ serve as the imputation model (fixed, no factor model).
WOR-active uses min(Ŷ, 1-Ŷ) scoring and the WOR-PAI estimator (eq. 2).
Bernoulli-active uses independent Bernoulli draws with the same scoring.

---

## File Structure

```
alphafold_study/
├── README.md
├── download_data.py     # download and preprocess data (requires gdown)
├── run_wor.py           # WOR-active + WOR-uniform (replaces run_wor.py)
├── run_bernoulli.py     # Bernoulli-active + Bernoulli-uniform + Classical
├── analyze_results.py   # combine CSVs, compute ESS multipliers, save summary
├── plot_results.py      # generate all 4 figures
├── submit.sh            # SLURM: 6 jobs (2 scripts × 3 seed chunks)
├── analysis.ipynb       # run analyze + plot interactively from results/
└── results/
    ├── wor_sl=*.csv                          # WOR raw results (3 files)
    ├── bernoulli_sl=*.csv                    # Bernoulli raw results (3 files)
    ├── all_per_seed.csv                      # combined per-seed data
    ├── summary.csv                           # per-(group, budget, method) summary
    ├── alphafold_main.pdf                    # main figure (3 methods)
    ├── alphafold_main_sparse.pdf             # main figure, sparse budgets
    ├── alphafold_ess+coverage_vs_classical.pdf
    └── alphafold_ess+coverage_vs_wor_uniform.pdf
```

---

## Quick Start

Open `analysis.ipynb` — all results are precomputed in `results/`, no cluster needed.

## Get the Data (only needed to re-run experiments)

```bash
pip install gdown
python download_data.py   # saves data/Y.npy, data/Yhat.npy, data/Z.npy
```

## Re-run on a Cluster (optional)

```bash
sbatch submit.sh                  # 6 parallel jobs
python analyze_results.py         # combine and summarize
python plot_results.py            # generate figures
```

## Dependencies

```bash
conda activate faq_env
```

Required: `torch`, `numpy`, `pandas`, `scipy`, `matplotlib`.
