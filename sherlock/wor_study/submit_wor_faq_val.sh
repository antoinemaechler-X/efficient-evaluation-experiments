#!/bin/bash
#SBATCH --job-name=wor_val
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/val/slurm_wor_val_%A_%a.out
#SBATCH --error=logs/val/slurm_wor_val_%A_%a.err
#SBATCH --array=0-9

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd ~/efficiently-evaluating-llms

# 10 jobs: 2 datasets x 5 gammas
dataset=$((SLURM_ARRAY_TASK_ID / 5))
gamma=$((SLURM_ARRAY_TASK_ID % 5))

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$dataset gamma=$gamma"
python wor_faq_val.py $dataset $gamma
