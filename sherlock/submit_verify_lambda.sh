#!/bin/bash
#SBATCH --job-name=verify_lambda
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm_verify_lambda_%A_%a.out
#SBATCH --error=logs/slurm_verify_lambda_%A_%a.err
#SBATCH --array=0-19

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 20 jobs: 2 datasets x 10 budgets
DATASETS=("mmlu-pro" "bbh+gpqa+ifeval+math+musr")
BUDGETS=("0.025" "0.05" "0.075" "0.1" "0.125" "0.15" "0.175" "0.2" "0.225" "0.25")

dataset_idx=$((SLURM_ARRAY_TASK_ID / 10))
budget_idx=$((SLURM_ARRAY_TASK_ID % 10))

DATASET=${DATASETS[$dataset_idx]}
BUDGET=${BUDGETS[$budget_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET budget=$BUDGET"
python verify_lambda.py --dataset "$DATASET" --budget "$BUDGET" --n_seeds 500 --save_profile
