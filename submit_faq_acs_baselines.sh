#!/bin/sh
#SBATCH --job-name=faq_acs_base
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --output=logs/acs/base_slurm_%A_%a.out
#SBATCH --error=logs/acs/base_slurm_%A_%a.err
#SBATCH --array=0-2

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

Ds=(8 16 32)
D=${Ds[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: D=$D (all estimators)"
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D $D --estimators faq,classical,uniform+pai --out_csv faq_acs_D${D}_all.csv --out_plot /dev/null

echo "Plotting..."
python plot_paper.py faq_acs_D${D}_all.csv --out faq_acs_D${D}_figure2.png
echo "Done. Figure saved to acs_study/faq_acs_D${D}_figure2.png"
