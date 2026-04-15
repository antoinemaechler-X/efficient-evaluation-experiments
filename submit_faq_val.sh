#!/bin/sh
#SBATCH --job-name=faq_val
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/val/slurm_%A_%a.out
#SBATCH --error=logs/val/slurm_%A_%a.err
#SBATCH --array=0-19

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

# 20 jobs: 2 datasets x 5 gammas x 2 missingness sets
dataset=$((SLURM_ARRAY_TASK_ID / 10))
remainder=$((SLURM_ARRAY_TASK_ID % 10))
gamma=$((remainder / 2))
ms=$((remainder % 2))

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$dataset gamma=$gamma ms=$ms"
python faq_val.py $dataset $gamma $ms
