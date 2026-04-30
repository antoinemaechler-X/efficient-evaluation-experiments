#!/bin/bash
#SBATCH --job-name=faq_acs
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=08:00:00
#SBATCH --output=logs/acs/slurm_%A_%a.out
#SBATCH --error=logs/acs/slurm_%A_%a.err
#SBATCH --array=0-2

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments/acs_study

mkdir -p ../logs/acs

# 3 SVD dims to compare: D in {8, 16, 32}
Ds=(8 16 32)
D=${Ds[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: D=$D"
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D $D --out_csv faq_acs_D${D}.csv --out_plot faq_acs_D${D}.png
