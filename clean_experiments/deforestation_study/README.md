# Amazon Deforestation Study

WOR (our method) vs. Cross-PPI (Zrnic & Candes, 2024), Zrnic's active inference, and Bernoulli
active inference, plus classical estimation, on the Amazon deforestation dataset from the
cross-ppi repo.

## Dataset

Parcels in the Amazon with a binary deforestation label (any disturbance between 2000 and 2015,
from Bullock et al. 2020) and two canopy-cover features (2015 and 2000, from Sexton et al. 2013).
The estimand is the fraction of parcels deforested. We follow Zrnic & Candes (2024): a
HistGradientBoostingClassifier gives the predictions, theta* is the mean over all parcels.

- N = 3,192 parcels total (theta* = 0.154)
- `X_all.npy`, `Y_all.npy`: full dataset, used by Cross-PPI's K-fold cross-fitting
- `Y_test.npy`, `Yhat_test.npy`: 50/50 test half (N=1,596), the inference pool for WOR / Bernoulli / active inference
- `Y_train2.npy`, `Yhat_train2.npy`: held-out tuning set for active inference
- `theta_true.npy`: theta* = mean(Y) over all parcels

`download_data.py` fetches the raw `data.csv` from the cross-ppi repo and regenerates every array.

## Experiments

Single population, 1000 seeds, 20 budgets (1% to 20% of N).

- `run_wor.py`: WOR estimator, `wor-active` (scoring by min(Yhat, 1-Yhat), tau=0.5) and `wor-uniform`.
- `run_bernoulli.py`: Bernoulli baselines `bernoulli-active`, `bernoulli-uniform`, `classical`.
- `run_cross_ppi.py`: Cross-PPI with K=10 cross-fitting and B=30 bootstrap. CPU-only and slow
  (it refits the model for every fold and bootstrap sample).
- `run_active_inference.py`: Zrnic's active inference with per-budget tau tuning on the held-out set.

### Status

Nothing has run yet, so `results/` is empty and there is no summary or figures. The scripts and
data are ready; running `submit.sh` on Marlowe and then the analysis and plot steps produces
everything. Both the analysis and the notebook handle partial results, so methods show up as they
finish rather than all at once.

## Usage

To run the experiments on Marlowe:

```bash
python download_data.py   # only if data/ is empty
sbatch submit.sh
```

Then build the summary and figures:

```bash
python analyze_results.py   # raw CSVs -> all_per_seed.csv, summary.csv
python plot_results.py      # summary.csv -> figures
```

`analysis.ipynb` reads whatever is in `results/` and shows the comparison once results exist.

## Layout

```
deforestation_study/
├── download_data.py
├── run_wor.py
├── run_bernoulli.py
├── run_cross_ppi.py
├── run_active_inference.py
├── analyze_results.py        # raw CSVs -> all_per_seed.csv, summary.csv
├── plot_results.py           # summary.csv -> figures
├── submit.sh
├── analysis.ipynb
├── environment.yml
├── data/                     # X_all, Y_all, Y_test, Yhat_test, Y_train2, Yhat_train2, theta_true
└── results/                  # raw CSVs, summaries, and figures land here (empty until the runs finish)
```

## Dependencies

See `environment.yml`: torch, numpy, pandas, scipy, matplotlib, scikit-learn (Cross-PPI and the
data prep use HistGradientBoostingClassifier).
