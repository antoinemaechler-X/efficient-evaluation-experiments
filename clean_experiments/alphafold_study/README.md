# AlphaFold Study

WOR (our method) vs. Bernoulli active inference (Zrnic & Candes, 2024) vs. classical
estimation, on the AlphaFold proteomics dataset from ppi_py.

## Dataset

10,802 proteins, each with:
- `Y`: binary disorder label
- `Yhat`: AlphaFold predicted disorder probability
- `Z`: phosphorylation flag

We split on `Z` and estimate the mean disorder rate in each group separately:
- Non-phosphorylated (Z=0): ~4,785 proteins
- Phosphorylated (Z=1): ~6,017 proteins

## Experiments

Mean estimation per group, over 1000 seeds and 20 budgets (1% to 20% of group size).

- `run_wor.py`: WOR estimator, `wor-active`
  (scoring by min(Yhat, 1-Yhat), tau=0.5) and `wor-uniform`.
- `run_bernoulli.py`: Bernoulli baselines `bernoulli-active`, `bernoulli-uniform`, and
  `classical` (no predictions, f=0).

Each script splits the 1000 seeds into 3 chunks for parallel cluster runs. `submit.sh`
launches all 6 jobs (2 scripts x 3 chunks).

## Usage

Results are precomputed under `results/`, so the notebook runs without a cluster:

```bash
jupyter notebook analysis.ipynb
```

To rebuild the summary and figures from the raw per-seed CSVs:

```bash
python analyze_results.py   # raw CSVs -> all_per_seed.csv, summary.csv
python plot_results.py      # summary.csv -> figures
```

To rerun the experiments from scratch on Marlowe:

```bash
pip install gdown && python download_data.py   # only if data/ is empty
sbatch submit.sh
```

## Layout

```
alphafold_study/
├── download_data.py
├── run_wor.py
├── run_bernoulli.py
├── analyze_results.py        # raw CSVs -> all_per_seed.csv, summary.csv
├── plot_results.py           # summary.csv -> figures
├── submit.sh
├── analysis.ipynb
├── environment.yml
├── data/                     # Y.npy, Yhat.npy, Z.npy
└── results/                  # raw CSVs, summaries, and figures
    ├── wor_sl={0,1,2}.csv
    ├── bernoulli_sl={0,1,2}.csv
    ├── all_per_seed.csv
    ├── summary.csv
    └── alphafold_*.pdf
```

## Dependencies

See `environment.yml`: torch, numpy, pandas, scipy, matplotlib.
