#!/bin/bash
#SBATCH --job-name=af_fix
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=alphafold_study/logs/slurm_af_fix_%A_%a.out
#SBATCH --error=alphafold_study/logs/slurm_af_fix_%A_%a.err
#SBATCH --array=0-2

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 3 jobs: run_bernoulli.py × 3 seed chunks (fixed: no clipping, ddof=0)
# NOTE: run_wor.py is NOT re-run (no changes)

SEED_CHUNK=$SLURM_ARRAY_TASK_ID

echo "Task $SLURM_ARRAY_TASK_ID: run_bernoulli.py seed_chunk=$SEED_CHUNK"
python alphafold_study/run_bernoulli.py $SEED_CHUNK
