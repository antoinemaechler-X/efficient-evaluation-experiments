# Clean Experiments

Reproducible packages for the four real-world binary-outcome studies in the paper. Each one
applies our WOR method to a dataset where predictions are available, and compares it against the
relevant baselines (Bernoulli active inference, Zrnic's active inference, Cross-PPI, and classical
estimation). Every study is self-contained: data, scripts, precomputed results, and a notebook.

## The four studies

| Study | Dataset | Estimand | Methods | Status |
|-------|---------|----------|---------|--------|
| `alphafold_study` | AlphaFold proteomics (N=10,802) | mean disorder rate, per phosphorylation group | WOR, Bernoulli, classical | complete |
| `galaxy_study` | Galaxy Zoo 2 (N=16,743) | spiral fraction | WOR, Bernoulli, Cross-PPI, classical | complete; active inference pending |
| `pew_study` | Pew ATP Wave 79 (N=6,189 test) | Biden approval rate | WOR, Bernoulli, active inference, classical | Bernoulli done; WOR and active inference pending |
| `deforestation_study` | Amazon deforestation (N=3,192) | fraction deforested | WOR, Bernoulli, Cross-PPI, active inference, classical | all pending |

All four estimate a binary mean over 1000 seeds and 20 budgets (1% to 20% of the pool), and report
two things per method: effective sample size relative to classical, and CI coverage.

## Methods

- **WOR** (ours): sample without replacement, `wor-active` scores items by min(Yhat, 1-Yhat) with
  tau=0.5, `wor-uniform` samples uniformly.
- **Bernoulli**: independent Bernoulli sampling (Zrnic & Candes style), `bernoulli-active`,
  `bernoulli-uniform`, and `classical` (no predictions).
- **Active inference** (Zrnic & Candes): Bernoulli sampling with tau tuned per budget on a held-out set.
- **Cross-PPI** (Zrnic & Candes): K-fold cross-fitting with bootstrap variance, on the full dataset.

## Layout

Every study follows the same structure:

```
<study>/
├── README.md            # what it studies, how to run it, current status
├── download_data.py     # fetch and preprocess the data
├── run_*.py             # one script per method (each splits 1000 seeds into 3 chunks)
├── analyze_results.py   # raw per-seed CSVs -> all_per_seed.csv, summary.csv
├── plot_results.py      # summary.csv -> figures
├── submit.sh            # SLURM array job for Marlowe
├── analysis.ipynb       # explore the precomputed results without a cluster
├── environment.yml
├── data/                # input arrays
└── results/             # raw CSVs, summaries, and figures
```

The analysis and plotting steps tolerate partial results, so methods that are still pending simply
do not appear yet, and fill in once their runs finish.

## Getting started

Open any study's `analysis.ipynb` to see its results (those that exist are precomputed, no cluster
needed). To rerun an experiment, follow the Usage section in that study's README.
