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

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$SLURM_ARRAY_TASK_ID"
python active_inference_factor_ablation.py $SLURM_ARRAY_TASK_ID
