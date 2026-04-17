#!/bin/sh
#SBATCH --job-name=faq_acs_clean
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --output=logs/acs/clean_%j.out
#SBATCH --error=logs/acs/clean_%j.err

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

echo "=== ACS FAQ clean run (D=16, all estimators) ==="
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D 16 --estimators faq,classical,uniform+pai --out_csv faq_acs_D16_clean.csv --out_plot faq_acs_D16_clean.png

echo "Plotting paper-style figure..."
python plot_paper.py faq_acs_D16_clean.csv --out faq_acs_D16_figure2.png
echo "=== Done ==="
