#!/bin/bash
#SBATCH --job-name=galaxy_fix
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=galaxy_study/logs/slurm_galaxy_fix_%A_%a.out
#SBATCH --error=galaxy_study/logs/slurm_galaxy_fix_%A_%a.err
#SBATCH --array=0-5

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 6 jobs: 2 scripts × 3 seed chunks
# 0-2: run_bernoulli.py (fixed: no clipping, ddof=0)
# 3-5: run_cross_ppi.py (fixed: no clipping)
# NOTE: run_wor.py is NOT re-run (no changes)

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_bernoulli.py seed_chunk=$SEED_CHUNK"
    python galaxy_study/run_bernoulli.py $SEED_CHUNK
else
    echo "Task $SLURM_ARRAY_TASK_ID: run_cross_ppi.py seed_chunk=$SEED_CHUNK"
    python galaxy_study/run_cross_ppi.py $SEED_CHUNK
fi
