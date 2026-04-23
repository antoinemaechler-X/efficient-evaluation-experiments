#!/bin/sh
#SBATCH --job-name=wor_val
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/val/slurm_wor_val_%A_%a.out
#SBATCH --error=logs/val/slurm_wor_val_%A_%a.err
#SBATCH --array=0-9

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

# 10 jobs: 2 datasets x 5 gammas
dataset=$((SLURM_ARRAY_TASK_ID / 5))
gamma=$((SLURM_ARRAY_TASK_ID % 5))

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$dataset gamma=$gamma"
python wor_faq_val.py $dataset $gamma
