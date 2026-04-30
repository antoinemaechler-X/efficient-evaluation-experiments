#!/bin/bash
#SBATCH --job-name=faq_acs_lowlabel
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=06:00:00
#SBATCH --output=logs/acs/lowlabel_%A_%a.out
#SBATCH --error=logs/acs/lowlabel_%A_%a.err
#SBATCH --array=0-2

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments/acs_study

mkdir -p ../logs/acs

N_LABS=(100 500 2000)
N_LAB=${N_LABS[$SLURM_ARRAY_TASK_ID]}

echo "=== ACS FAQ low-label run: n_labeled=$N_LAB ==="
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D 16 --n_labeled $N_LAB --estimators faq,classical,uniform+pai --out_csv faq_acs_nlab${N_LAB}.csv --out_plot faq_acs_nlab${N_LAB}.png

echo "Plotting..."
python plot_paper.py faq_acs_nlab${N_LAB}.csv --out faq_acs_nlab${N_LAB}_figure2.png
echo "=== Done: n_labeled=$N_LAB ==="
