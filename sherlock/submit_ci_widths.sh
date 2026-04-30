#!/bin/bash
#SBATCH --job-name=ci_widths
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/ci_widths/slurm_%A_%a.out
#SBATCH --error=logs/ci_widths/slurm_%A_%a.err
#SBATCH --array=0-2

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

echo "Task $SLURM_ARRAY_TASK_ID: seed_chunk=$SLURM_ARRAY_TASK_ID"
python faq_final_ci_saved.py $SLURM_ARRAY_TASK_ID
