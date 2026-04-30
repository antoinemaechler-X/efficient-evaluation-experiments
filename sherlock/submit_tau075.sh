#!/bin/bash
#SBATCH --job-name=faq_tau075
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=logs/tau075/slurm_%A_%a.out
#SBATCH --error=logs/tau075/slurm_%A_%a.err
#SBATCH --array=0-1

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$SLURM_ARRAY_TASK_ID"
python faq_tau075.py $SLURM_ARRAY_TASK_ID
