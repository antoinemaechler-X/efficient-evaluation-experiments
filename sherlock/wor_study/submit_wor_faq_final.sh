#!/bin/bash
#SBATCH --job-name=wor_final
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_wor_final_%A_%a.out
#SBATCH --error=logs/final/slurm_wor_final_%A_%a.err
#SBATCH --array=0-2

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 3 jobs: seed chunks 0-2
echo "Task $SLURM_ARRAY_TASK_ID: seed_chunk=$SLURM_ARRAY_TASK_ID"
python wor_faq_final.py $SLURM_ARRAY_TASK_ID
