#!/bin/sh
#SBATCH --job-name=faq_acs_nlab50
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --output=logs/acs/nlab50_%j.out
#SBATCH --error=logs/acs/nlab50_%j.err

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

echo "=== ACS FAQ n_labeled=50 (ratio~0.48, max FAQ signal) ==="
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D 16 --n_labeled 50 --estimators faq,classical,uniform+pai --out_csv faq_acs_nlab50.csv --out_plot faq_acs_nlab50.png

echo "Plotting..."
python plot_paper.py faq_acs_nlab50.csv --out faq_acs_nlab50_figure2.png
echo "=== Done ==="
