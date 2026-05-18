# Deforestation Study — Task Description

## Goal

Reproduce the deforestation experiment from the **Cross-PPI** paper (Zrnic & Candes, 2024, PNAS) and compare their method against our **WOR-AIPW** (FAQ) method. This follows the same pattern as `galaxy_study/` (which compared against Cross-PPI) and `pew_study/` (which compared against Active Inference).

**Source repo**: https://github.com/tijana-zrnic/cross-ppi
**Notebook**: `deforestation/deforestation.ipynb`
**Data file**: `deforestation/data.csv` (522 KB, 3,193 rows)

---

## The Dataset

### Source
- **Labels**: Bullock et al. (2020), "Satellite-based estimates reveal widespread forest degradation in the Amazon," *Global Change Biology*. Gold-standard deforestation labels from field visits to Amazon parcels. Original at https://github.com/bullocke/amazon (`reference_samples.xlsx`).
- **Features**: Sexton et al. (2013), GLCF tree canopy cover product from Google Earth Engine at 30m resolution.

### Format
CSV with 3,193 rows, 15 columns. After filtering 1 bad row (`Year1 = 'tdo'`): **N = 3,192**.

### Preprocessing (exactly as Zrnic)
```python
raw_df = pd.read_csv('./data.csv')
df = raw_df[raw_df.Year1 != 'tdo'].copy()
df.Year1 = df.Year1.astype(float)

# Y: binary deforestation indicator (any disturbance event between 2000-2015)
Y_all = (
    ((df.Type1.astype(str) != 'nan') & ((df.Year1 >= 2000) & (df.Year1 <= 2015))) |
    ((df.Type2.astype(str) != 'nan') & ((df.Year2 >= 2000) & (df.Year2 <= 2015))) |
    ((df.Type3.astype(str) != 'nan') & ((df.Year3 >= 2000) & (df.Year3 <= 2015)))
).astype(float).to_numpy()

# X: 2D features (tree canopy cover in 2015 and 2000, scaled to [0,1])
X_all = np.stack([
    df.tree_canopy_cover_2015 / 100,
    df.tree_canopy_cover_2000 / 100
], axis=1)
```

### Key statistics
| | Value |
|---|---|
| Total parcels | 3,192 |
| Deforested (Y=1) | 492 (15.4%) |
| Not deforested (Y=0) | 2,700 (84.6%) |
| Features | 2D: canopy cover 2015 and 2000 |
| Estimand | theta* = mean(Y_all) = fraction deforested |

---

## Zrnic's Cross-PPI Method (to reproduce exactly)

### Model
`HistGradientBoostingClassifier(max_iter=100, max_depth=2)` from scikit-learn.
Predictions: `cls.predict_proba(X)[:, 1]`.

### Cross-PPI estimator (K=10 fold cross-fitting)
For each trial:
1. Randomly split all 3,192 items into `n` labeled and `N-n` unlabeled
2. K-fold cross-fit on labeled data:
   - For each fold j: train model on K-1 folds, predict on held-out fold → `Yhat_labeled[fold_j]`
   - For each fold j: predict on all unlabeled items, accumulate average → `Yhat_unlabeled += preds / K`
3. Point estimate: `theta_hat = mean(Yhat_unlabeled) + mean(Y_labeled - Yhat_labeled)`
4. Variance: bootstrap-based (B=30 resamples), see below
5. CI: `theta_hat +/- z_{1-alpha/2} * sqrt(var_hat)`

### Bootstrap variance (B=30)
```python
For b in 1..B:
    Subsample train_n = n - n/K labeled items (with replacement from labeled)
    Train model, predict on unlabeled → accumulate Yhat_unlabeled_avg
    Predict on out-of-sample labeled → collect residuals
var_hat = var(Yhat_unlabeled_avg)/N + var(residuals)/n
```

### PPI with data splitting (second baseline)
- Split labeled: n_tr = int(0.1 * n) for training, rest for debiasing
- Train single model on n_tr items
- `theta_PPI = mean(Yhat_unlabeled) + mean(Y_val - Yhat_val)`
- `var = var(Yhat_unlabeled)/N + var(Y_val - Yhat_val)/(n - n_tr)`

### Classical
- `theta_hat = mean(Y_labeled)`
- `var = var(Y_labeled) / n`

### Zrnic's parameters
| Parameter | Value |
|-----------|-------|
| Budget fractions | 0.1, 0.2, 0.3 |
| Labeled sizes n | 319, 638, 957 |
| num_trials | 100 |
| alpha | 0.1 (90% CI) |
| K (cross-fitting folds) | 10 |
| Bootstrap B | 30 |

---

## What to Implement

Follow the EXACT same folder organization as `galaxy_study/` and `pew_study/`. Read those folders carefully first.

### Scripts to create

1. **`download_data.py`**
   - Download `data.csv` from the cross-ppi GitHub repo
   - Preprocess exactly as Zrnic (filtering, binarization, feature extraction)
   - Train/test split (50/50) — train model on training half, predict on test half
   - Save `Y_test.npy`, `Yhat_test.npy`, `theta_true.npy` (= mean(Y_all))
   - Also save `X_test.npy`, `X_train.npy`, `Y_train.npy` (needed for Cross-PPI cross-fitting)
   - Model: same `HistGradientBoostingClassifier(max_iter=100, max_depth=2)`

2. **`run_wor.py`** — WOR-Active + WOR-Uniform (our methods, GPU-vectorized)
   - Uses fixed Yhat_test predictions (from model trained on full training data)
   - Identical structure to `galaxy_study/run_wor.py` and `pew_study/run_wor.py`

3. **`run_bernoulli.py`** — Bernoulli-Active + Bernoulli-Uniform + Classical
   - Uses fixed Yhat_test predictions
   - Identical structure to `galaxy_study/run_bernoulli.py` and `pew_study/run_bernoulli.py`

4. **`run_cross_ppi.py`** — Cross-PPI reproduction (Zrnic's method)
   - **This is the tricky one**: Cross-PPI retrains the model K=10 times per trial via cross-fitting
   - For each trial at each budget: randomly select n items as labeled, run K-fold cross-fitting, compute Cross-PPI estimator with bootstrap variance
   - Must reproduce Zrnic's exact code (K=10, B=30, HistGradientBoostingClassifier)
   - Unlike galaxy/pew where baseline was simple, this requires training models INSIDE the trial loop
   - CPU-only (scikit-learn, no GPU needed for this script)

5. **`analyze_results.py`** — Merge CSVs, compute ESS multipliers (same pattern as galaxy/pew)

6. **`plot_results.py`** — Generate figures (same pattern as galaxy/pew)

7. **`submit.sh`** (Marlowe) + **`sherlock/deforestation_study/submit.sh`** (Sherlock)
   - Array jobs: 0-8 (3 scripts x 3 seed chunks)

### Key design decisions

- **Our methods (WOR, Bernoulli)** use FIXED predictions from a model trained on 50% of data, predicting on the other 50%. This is the same approach as `pew_study/`.
- **Cross-PPI** operates on the FULL dataset (all 3,192 items) with its own K-fold cross-fitting, exactly as Zrnic does. For each trial, it randomly selects n items as "labeled" from the full dataset and cross-fits on those.
- **theta_true = mean(Y_all)** over all 3,192 items (same as Zrnic).
- **alpha = 0.05** (our standard, not Zrnic's 0.1).
- **Budget proportions**: 1% to 20% in 20 steps (our standard range; Zrnic only uses 10%, 20%, 30%).
- **1,000 seeds** split into 3 chunks (our standard; Zrnic uses only 100 trials).
- **Scoring**: `s_j = min(Yhat_j, 1-Yhat_j)` for active methods, `tau = 0.5`.

### Important note on Cross-PPI implementation

Cross-PPI is computationally expensive because it trains K=10 models per trial x 1000 seeds x 20 budgets = 200,000 model fits. Consider:
- Using `n_jobs=-1` in HistGradientBoostingClassifier for parallelism
- Running fewer seeds if needed (e.g., 100 trials like Zrnic)
- The cross_ppi script should be CPU-only (no GPU needed)
- May need longer wall time than WOR/Bernoulli scripts

### Reference files
- `galaxy_study/` — template for folder organization, run scripts, analysis, plotting
- `pew_study/` — template for train/test split approach with model training
- `galaxy_study/run_cross_ppi.py` — simple Cross-PPI (without cross-fitting, galaxy had pre-computed predictions)
- Cross-PPI notebook: https://github.com/tijana-zrnic/cross-ppi/blob/main/deforestation/deforestation.ipynb

### Output CSV format (same as galaxy/pew)
Each row: `group, prop_budget, method, seed, width, coverage`

Methods: `wor-active`, `wor-uniform`, `bernoulli-active`, `bernoulli-uniform`, `classical`, `cross-ppi`
