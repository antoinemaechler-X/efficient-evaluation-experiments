Precise Summary of ACS Study Work                                                                                                                                                                                                                                                     
  Goal
                                                                                                                                                
  Reproduce the FAQ/PAI experiments (originally on MMLU/GPQA benchmarks) with the ACS Census 2019 California dataset. Target: estimate the OLS
  coefficient of AGEP (age) when regressing PINCP (personal income) on (AGEP, SEX) — using adaptive sampling to get valid confidence intervals  
  with minimal labels.                                                                                                                        

  Files created in acs_study/

  1. utils.py

  Adapted from tijana-zrnic/active-inference. Three functions:
  - get_data(year, features, outcome) — loads ACS CA data from local CSV (acs_study/data/2019/1-Year/psam_p06.csv), falls back to folktables
  download if missing. Reads only needed columns to avoid OOM.
  - transform_features(features, ft, enc=None) — one-hot encodes categoricals, keeps quantitatives. Returns sparse CSC matrix.
  - ols(features, outcome) — OLS via pseudo-inverse.

  2. train_model.py

  Trains two XGBoost regressors (reg:absoluteerror / median regression), mirroring the census-analysis notebook:
  - Primary model: predicts PINCP from 17 demographic features.
  - Error model: predicts |y - ŷ| — used as per-point uncertainty.

  Saves 4 artefacts (all needed for FAQ):
  - <stem>.npz — Y, Yhat, predicted_errs, X (AGE,SEX for unlabeled), theta_true
  - <stem>.tree.json — XGBoost booster (for fine-tuning)
  - <stem>.tree_err.json — error booster
  - <stem>.features.npz — encoded unlabeled feature matrix (sparse)

  Key args: --n_rounds (200 quick / 2000 full), --n_labeled (absolute number of training labels, overrides --train_frac), --seed.

  3. run_baseline.py — VERIFIED WORKING

  Reproduces census-analysis.ipynb (one-shot active inference, no fine-tuning). Three estimators compared across budgets:
  - active — Bernoulli sampling with probs ∝ uncertainty, AIPW estimator
  - uniform — Bernoulli sampling at fixed budget rate, AIPW estimator
  - classical — no ML predictions, just Horvitz-Thompson

  Uses sandwich variance for CIs. Default: 11 budgets from 0.5% to 10%, target coverage 90%.

  4. run_faq.py — NOT YET TESTED

  Sequential PAI/FAQ, adapted from census-analysis-sequential.ipynb. Per trial:
  - Random permutation of the unlabeled stream
  - Iterate point-by-point; at each step compute adaptive sampling prob (mixes uncertainty-weighted + uniform via tau, with a "greedy terminal
  samples" mechanism to exhaust remaining budget)
  - After every batch_size labels, fine-tune the XGBoost model on the newly-labeled batch → updates Yhat and predicted_errs for future sampling
  - Also run a coupled "no fine-tuning" version + uniform for comparison

  Three estimators: active (w/ fine-tuning), active (no fine-tuning), uniform.

  5. check_data.py, load_data.py — data verification

  6. environment_local.yml

  Local conda env (acs_env), mirrors cluster faq_env except drops linux/CUDA packages. Contains: numpy, scipy, pandas, scikit-learn, matplotlib,
   seaborn, xgboost, folktables, tqdm.

  Files at repo root

  - GIT_WORKFLOW.md — reference doc for local↔cluster workflow
  - .gitignore — excludes big data/.npz/experiment logs

  Git setup (complete)

  - origin → https://github.com/antoinemaechler-X/efficient-evaluation-experiments.git (user's repo)
  - upstream → skbwu's original repo (unchanged)
  - Both local and cluster configured the same way, main tracking origin/main.

  Status

  Verified locally:
  - Data loads correctly (380,091 CA rows)
  - train_model.py with 200 rounds produces predictions_test.npz
  - train_model.py with 2000 rounds produced full predictions.npz
  - run_baseline.py ran successfully (100 trials × 10 budgets) — active beats uniform beats classical as expected, ~90% coverage everywhere ✓

  Just added, not yet tested:
  - run_faq.py (sequential PAI with fine-tuning)
  - Updated train_model.py to save models + encoded features

  Immediate next step (what the user should run)

  Since old predictions_test.npz lacks the new companion files (.tree.json, .tree_err.json, .features.npz), re-run train first, then FAQ smoke
  test:

  cd ~/efficiently-evaluating-llms/acs_study

  python train_model.py --n_rounds 200 --n_labeled 400 --out predictions_test.npz

  python run_faq.py --predictions predictions_test.npz \
      --num_trials 3 --num_budgets 3 --budget_min 0.01 --budget_max 0.05 \
      --batch_size 500 --ft_steps 100 --greedy_steps 200 \
      --n_max 20000 \
      --out_csv faq_test.csv --out_plot faq_test.png

  Should finish in a few minutes. If it works:
  1. Push everything to GitHub
  2. Pull on cluster
  3. Check faq_env has xgboost/folktables/seaborn; pip install any missing
  4. Re-run train_model.py on cluster (or scp artefacts), then submit FAQ as sbatch job

  PhD student's instructions (applied)

  - Budgets from 0.5% to 10% ✓ (11 evenly-spaced values)
  - "0.1% training labels" we add this bidget to see.

  Open / unfinished

  - FAQ smoke test not run yet
  - run_faq.py has a Python per-point inner loop over N≈190k — this is inherently slow. Local testing uses --n_max to truncate. Full cluster run
   will take hours.
  - Cluster env faq_env may need pip install xgboost folktables seaborn — user should check before submitting sbatch.