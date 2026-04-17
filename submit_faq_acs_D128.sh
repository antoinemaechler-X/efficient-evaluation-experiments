#!/bin/sh
#SBATCH --job-name=faq_acs_D128
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=10:00:00
#SBATCH --output=logs/acs/D128_%A_%a.out
#SBATCH --error=logs/acs/D128_%A_%a.err
#SBATCH --array=0-3

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/acs_study

mkdir -p ../logs/acs

# 0: full labels  (shows R² improvement, FAQ likely still = uniform+pai)
# 1: 500 labels   (very uncertain posterior, FAQ should clearly win)
# 2: 2000 labels  (moderate uncertainty)
# 3: 10000 labels (mild uncertainty, transition regime)
N_LABS=(0 500 2000 10000)
N_LAB=${N_LABS[$SLURM_ARRAY_TASK_ID]}

if [ "$N_LAB" -eq 0 ]; then
    LABEL_ARG="--train_frac 0.5"
    SUFFIX="fulllabel"
else
    LABEL_ARG="--n_labeled $N_LAB"
    SUFFIX="nlab${N_LAB}"
fi

echo "=== ACS FAQ D=128, $SUFFIX ==="
python run_faq.py --num_trials 100 --num_budgets 11 --budget_min 0.005 --budget_max 0.10 --D 128 $LABEL_ARG --estimators faq,classical,uniform+pai --out_csv faq_acs_D128_${SUFFIX}.csv --out_plot faq_acs_D128_${SUFFIX}.png

echo "Plotting..."
python plot_paper.py faq_acs_D128_${SUFFIX}.csv --out faq_acs_D128_${SUFFIX}_figure2.png
echo "=== Done: D=128, $SUFFIX ==="
