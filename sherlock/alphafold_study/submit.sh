#!/bin/bash
#SBATCH --job-name=alphafold
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=alphafold_study/logs/slurm_alphafold_%A_%a.out
#SBATCH --error=alphafold_study/logs/slurm_alphafold_%A_%a.err
#SBATCH --array=0-5

# Load CUDA (check available versions with: ml spider cuda)
# ml cuda/12.1.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/groups/candes/maechler/faq_env

cd ~/efficiently-evaluating-llms

# 6 jobs: 2 scripts × 3 seed chunks
# 0-2: run_wor.py with seed_chunk 0,1,2
# 3-5: run_bernoulli.py with seed_chunk 0,1,2

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_wor.py seed_chunk=$SEED_CHUNK"
    python alphafold_study/run_wor.py $SEED_CHUNK
else
    echo "Task $SLURM_ARRAY_TASK_ID: run_bernoulli.py seed_chunk=$SEED_CHUNK"
    python alphafold_study/run_bernoulli.py $SEED_CHUNK
fi
