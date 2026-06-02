# Galaxy Zoo 2 Study

WOR (our method) vs. Cross-PPI (Zrnic & Candes, 2024), Bernoulli active inference,
and classical estimation, on the Galaxy Zoo 2 dataset from ppi_py.

## Dataset

16,743 galaxies, each with:
- `Y`: binary spiral label (1 = spiral, 0 = not)
- `Yhat`: predicted spiral probability from a pretrained ResNet50 (shipped by ppi_py)

We estimate the overall spiral fraction
(theta* = 0.2593). Mean prediction is 0.2603, so the model is close but slightly biased.

## Experiments

Single population, 1000 seeds, 20 budgets (1% to 20% of N).

- `run_wor.py`: WOR estimator, `wor-active`
  (scoring by min(Yhat, 1-Yhat), tau=0.5) and `wor-uniform`.
- `run_bernoulli.py`: Bernoulli baselines `bernoulli-active`, `bernoulli-uniform`, `classical`.
- `run_cross_ppi.py`: Cross-PPI estimator from Zrnic & Candes (2024).
- `run_active_inference.py`: Zrnic's active inference with per-budget tau tuning. **Not run yet**
  (see Status), but the script is ready and the rest of the pipeline picks it up automatically
  once its results land in `results/`.

Each script splits the 1000 seeds into 3 chunks for parallel cluster runs. `submit.sh` launches
all 12 jobs (4 scripts x 3 chunks).

### Status

WOR, Bernoulli and Cross-PPI are complete. Active inference is still pending, so it does not yet
appear in `summary.csv` or the figures. Running it on Marlowe and re-running the analysis and plot
steps will add it everywhere.

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
pip install ppi-python && python download_data.py   # only if data/ is empty
sbatch submit.sh
```

## Layout

```
galaxy_study/
├── download_data.py
├── run_wor.py
├── run_bernoulli.py
├── run_cross_ppi.py
├── run_active_inference.py   # pending
├── analyze_results.py        # raw CSVs -> all_per_seed.csv, summary.csv
├── plot_results.py           # summary.csv -> figures
├── submit.sh
├── analysis.ipynb
├── environment.yml
├── data/                     # Y.npy, Yhat.npy
└── results/                  # raw CSVs, summaries, and figures
    ├── wor_sl={0,1,2}.csv
    ├── bernoulli_sl={0,1,2}.csv
    ├── cross_ppi_sl={0,1,2}.csv
    ├── all_per_seed.csv
    ├── summary.csv
    └── galaxy_*.pdf / .png
```

## Dependencies

See `environment.yml`: torch, numpy, pandas, scipy, matplotlib (and ppi-python to download data).
