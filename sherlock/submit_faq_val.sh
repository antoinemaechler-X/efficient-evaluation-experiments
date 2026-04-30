#!/bin/bash
#SBATCH --job-name=faq_val
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/val/slurm_%A_%a.out
#SBATCH --error=logs/val/slurm_%A_%a.err
#SBATCH --array=0-19

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd ~/efficiently-evaluating-llms

# 20 jobs: 2 datasets x 5 gammas x 2 missingness sets
dataset=$((SLURM_ARRAY_TASK_ID / 10))
remainder=$((SLURM_ARRAY_TASK_ID % 10))
gamma=$((remainder / 2))
ms=$((remainder % 2))

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$dataset gamma=$gamma ms=$ms"
python faq_val.py $dataset $gamma $ms
