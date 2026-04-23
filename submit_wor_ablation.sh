#!/bin/sh
#SBATCH --job-name=wor_abl
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_wor_abl_%A_%a.out
#SBATCH --error=logs/final/slurm_wor_abl_%A_%a.err
#SBATCH --array=0-5

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

# 6 jobs: 2 datasets x 3 seed chunks
dataset=$((SLURM_ARRAY_TASK_ID / 3))
seed_chunk=$((SLURM_ARRAY_TASK_ID % 3))

echo "Task $SLURM_ARRAY_TASK_ID: dataset=$dataset seed_chunk=$seed_chunk"
python wor_ablation.py $dataset $seed_chunk
