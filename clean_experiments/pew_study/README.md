# Pew ATP Wave 79 Study

WOR (our method) vs. Bernoulli active inference and Zrnic's active inference, plus classical
estimation, on Pew's American Trends Panel Wave 79 (November 2020, post-election).

## Dataset

Survey responses from Pew ATP Wave 79. The estimand is the Biden post-election messaging
approval rate. We follow Zrnic & Candes (2024): a 50/50 train/test split, an XGBoost model
trained on the training half giving predictions `Yhat` on the test half, and a second model
trained on 80% of the training half for tau tuning. theta* is the mean over all data (not just
the test set), to match their setup.

- `Y_test.npy`, `Yhat_test.npy`: test-set labels and predictions (the inference pool, N=6,189)
- `Y_train2.npy`, `Yhat_train2.npy`: held-out tuning set used by active inference
- `theta_true.npy`: theta* = mean(Y) over all data

The raw SPSS file (`ATPW79.sav`) is not bundled here, it needs a free Pew registration.
`download_data.py` documents where to get it and regenerates the `.npy` arrays.

## Experiments

Single population, 1000 seeds, 20 budgets (1% to 20% of N).

- `run_bernoulli.py`: Bernoulli baselines `bernoulli-active`, `bernoulli-uniform`, `classical`.
- `run_wor.py`: WOR estimator, `wor-active` (scoring by min(Yhat, 1-Yhat), tau=0.5) and `wor-uniform`.
- `run_active_inference.py`: Zrnic's active inference with per-budget tau tuning on the held-out set.

### Status

Only Bernoulli has run so far. WOR and active inference are still pending, so they do not yet
appear in `summary.csv` or the figures, and the ESS-vs-WOR-uniform figure is skipped until WOR
exists. Running them on Marlowe and re-running the analysis and plot steps fills everything in.

## Usage

The Bernoulli results are precomputed under `results/`, so the notebook runs without a cluster:

```bash
jupyter notebook analysis.ipynb
```

To rebuild the summary and figures from the raw per-seed CSVs:

```bash
python analyze_results.py   # raw CSVs -> all_per_seed.csv, summary.csv
python plot_results.py      # summary.csv -> figures
```

To run the pending experiments on Marlowe:

```bash
# put ATPW79.sav in data/ first (see download_data.py), then:
python download_data.py
sbatch submit.sh
```

## Layout

```
pew_study/
├── download_data.py
├── run_bernoulli.py
├── run_wor.py                # pending
├── run_active_inference.py   # pending
├── analyze_results.py        # raw CSVs -> all_per_seed.csv, summary.csv
├── plot_results.py           # summary.csv -> figures
├── submit.sh
├── analysis.ipynb
├── environment.yml
├── data/                     # Y_test, Yhat_test, Y_train2, Yhat_train2, theta_true
└── results/                  # raw CSVs, summaries, and figures
    ├── bernoulli_sl={0,1,2}.csv
    ├── all_per_seed.csv
    ├── summary.csv
    └── pew_*.pdf / .png
```

## Dependencies

See `environment.yml`: torch, numpy, pandas, scipy, matplotlib (plus xgboost and pyreadstat to
rebuild the data).
