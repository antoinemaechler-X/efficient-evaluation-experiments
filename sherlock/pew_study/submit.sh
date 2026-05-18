#!/bin/bash
#SBATCH --job-name=pew
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=pew_study/logs/slurm_pew_%A_%a.out
#SBATCH --error=pew_study/logs/slurm_pew_%A_%a.err
#SBATCH --array=0-8

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 9 jobs: 3 scripts x 3 seed chunks
# 0-2: run_wor.py with seed_chunk 0,1,2
# 3-5: run_bernoulli.py with seed_chunk 0,1,2
# 6-8: run_active_inference.py with seed_chunk 0,1,2

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_wor.py seed_chunk=$SEED_CHUNK"
    python pew_study/run_wor.py $SEED_CHUNK
elif [ $SCRIPT_IDX -eq 1 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_bernoulli.py seed_chunk=$SEED_CHUNK"
    python pew_study/run_bernoulli.py $SEED_CHUNK
else
    echo "Task $SLURM_ARRAY_TASK_ID: run_active_inference.py seed_chunk=$SEED_CHUNK"
    python pew_study/run_active_inference.py $SEED_CHUNK
fi
