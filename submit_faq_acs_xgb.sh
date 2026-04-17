#!/bin/sh
#SBATCH --job-name=faq_acs_xgb
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH --output=logs/acs/xgb_%A_%a.out
#SBATCH --error=logs/acs/xgb_%A_%a.err
#SBATCH --array=0-3

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

# BLR initialised on N_BLR labeled points only.
# XGBoost always uses the full labeled split (~190k).
# 0: 200   — near-zero BLR knowledge, max FAQ signal
# 1: 1000  — low
# 2: 5000  — moderate
# 3: full  — BLR also on full split (baseline: shows XGB alone helps)
N_BLRS=(200 1000 5000 0)
N_BLR=${N_BLRS[$SLURM_ARRAY_TASK_ID]}

if [ "$N_BLR" -eq 0 ]; then
    BLR_ARG="--n_labeled_blr 190000"
    SUFFIX="xgb_fullblr"
else
    BLR_ARG="--n_labeled_blr $N_BLR"
    SUFFIX="xgb_nblr${N_BLR}"
fi

echo "=== ACS XGB+FAQ: $SUFFIX ==="
python run_faq_xgb.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D 16 $BLR_ARG --estimators faq,classical,uniform+pai --out_csv faq_acs_${SUFFIX}.csv --out_plot faq_acs_${SUFFIX}.png

echo "Plotting..."
python plot_paper.py faq_acs_${SUFFIX}.csv --out faq_acs_${SUFFIX}_figure2.png
echo "=== Done: $SUFFIX ==="
