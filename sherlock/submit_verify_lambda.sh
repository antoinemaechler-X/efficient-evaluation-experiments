#!/bin/bash
#SBATCH --job-name=verify_lambda
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_verify_lambda_%A_%a.out
#SBATCH --error=logs/slurm_verify_lambda_%A_%a.err
#SBATCH --array=0-3

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd ~/efficiently-evaluating-llms

# 4 jobs: 2 datasets x 2 budgets (0.1, 0.25)
DATASETS=("mmlu-pro" "bbh+gpqa+ifeval+math+musr")
BUDGETS=("0.1" "0.25")

dataset_idx=$((SLURM_ARRAY_TASK_ID / 2))
budget_idx=$((SLURM_ARRAY_TASK_ID % 2))

DATASET=${DATASETS[$dataset_idx]}
BUDGET=${BUDGETS[$budget_idx]}

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$DATASET budget=$BUDGET"
python verify_lambda.py --dataset "$DATASET" --budget "$BUDGET" --n_seeds 100
