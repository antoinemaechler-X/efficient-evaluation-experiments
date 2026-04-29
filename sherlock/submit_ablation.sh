#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_ablation_%A_%a.out
#SBATCH --error=logs/final/slurm_ablation_%A_%a.err
#SBATCH --array=0-1

# Load CUDA (check available versions with: ml spider cuda)
# ml cuda/12.1.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/groups/candes/maechler/faq_env

cd ~/efficiently-evaluating-llms

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$SLURM_ARRAY_TASK_ID"
python active_inference_factor_ablation.py $SLURM_ARRAY_TASK_ID
