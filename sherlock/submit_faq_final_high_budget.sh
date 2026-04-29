#!/bin/bash
#SBATCH --job-name=faq_final_hb
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_faq_final_hb_%A_%a.out
#SBATCH --error=logs/final/slurm_faq_final_hb_%A_%a.err
#SBATCH --array=0-2

# Load CUDA (check available versions with: ml spider cuda)
# ml cuda/12.1.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/groups/candes/maechler/faq_env

cd ~/efficiently-evaluating-llms

echo "Task $SLURM_ARRAY_TASK_ID: seed_chunk=$SLURM_ARRAY_TASK_ID"
python faq_final_high_budget_all_ms.py $SLURM_ARRAY_TASK_ID
