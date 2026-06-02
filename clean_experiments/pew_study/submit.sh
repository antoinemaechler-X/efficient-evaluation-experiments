#!/bin/sh
#SBATCH --job-name=pew
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_pew_%A_%a.out
#SBATCH --error=results/slurm_pew_%A_%a.err
#SBATCH --array=0-8

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/clean_experiments/pew_study

# 9 jobs: 3 scripts x 3 seed chunks
# 0-2: run_wor.py
# 3-5: run_bernoulli.py
# 6-8: run_active_inference.py

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    python run_wor.py $SEED_CHUNK
elif [ $SCRIPT_IDX -eq 1 ]; then
    python run_bernoulli.py $SEED_CHUNK
else
    python run_active_inference.py $SEED_CHUNK
fi
