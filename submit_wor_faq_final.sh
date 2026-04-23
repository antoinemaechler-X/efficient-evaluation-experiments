#!/bin/sh
#SBATCH --job-name=wor_final
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_wor_final_%A_%a.out
#SBATCH --error=logs/final/slurm_wor_final_%A_%a.err
#SBATCH --array=0-2

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

# 3 jobs: seed chunks 0-2
echo "Task $SLURM_ARRAY_TASK_ID: seed_chunk=$SLURM_ARRAY_TASK_ID"
python wor_faq_final.py $SLURM_ARRAY_TASK_ID
