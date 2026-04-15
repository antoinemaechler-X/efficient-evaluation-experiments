#!/bin/sh
#SBATCH --job-name=faq_tau075
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/tau075/slurm_%A_%a.out
#SBATCH --error=logs/tau075/slurm_%A_%a.err
#SBATCH --array=0-1

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$SLURM_ARRAY_TASK_ID"
python faq_tau075.py $SLURM_ARRAY_TASK_ID
