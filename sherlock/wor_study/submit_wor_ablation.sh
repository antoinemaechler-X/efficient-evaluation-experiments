#!/bin/bash
#SBATCH --job-name=wor_abl
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_wor_abl_%A_%a.out
#SBATCH --error=logs/final/slurm_wor_abl_%A_%a.err
#SBATCH --array=0-5

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 6 jobs: 2 datasets x 3 seed chunks
dataset=$((SLURM_ARRAY_TASK_ID / 3))
seed_chunk=$((SLURM_ARRAY_TASK_ID % 3))

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$dataset seed_chunk=$seed_chunk"
python wor_ablation.py $dataset $seed_chunk
