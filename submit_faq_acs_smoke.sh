#!/bin/sh
#SBATCH --job-name=faq_acs_smoke
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=00:15:00
#SBATCH --output=logs/acs/smoke_%j.out
#SBATCH --error=logs/acs/smoke_%j.err

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

echo "=== ACS FAQ smoke test ==="
python run_faq.py --num_trials 5 --num_budgets 3 --budget_min 0.025 --budget_max 0.10 --n_max 10000 --D 8 --out_csv faq_smoke.csv --out_plot /dev/null
python plot_paper.py faq_smoke.csv --out faq_smoke_figure2.pdf
echo "=== done ==="
