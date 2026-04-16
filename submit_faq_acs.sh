#!/bin/sh
#SBATCH --job-name=faq_acs
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=08:00:00
#SBATCH --output=logs/acs/slurm_%A_%a.out
#SBATCH --error=logs/acs/slurm_%A_%a.err
#SBATCH --array=0-2

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

# 3 SVD dims to compare: D in {8, 16, 32}
Ds=(8 16 32)
D=${Ds[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: D=$D"
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D $D --out_csv faq_acs_D${D}.csv --out_plot faq_acs_D${D}.png
