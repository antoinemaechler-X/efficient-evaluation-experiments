#!/bin/bash
#SBATCH --job-name=deforest
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=24:00:00
#SBATCH --output=deforestation_study/logs/slurm_deforest_%A_%a.out
#SBATCH --error=deforestation_study/logs/slurm_deforest_%A_%a.err
#SBATCH --array=0-11

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 12 jobs: 4 scripts x 3 seed chunks
# 0-2:  run_wor.py with seed_chunk 0,1,2              (GPU)
# 3-5:  run_bernoulli.py with seed_chunk 0,1,2         (GPU)
# 6-8:  run_cross_ppi.py with seed_chunk 0,1,2         (CPU-only, long runtime)
# 9-11: run_active_inference.py with seed_chunk 0,1,2   (GPU)

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_wor.py seed_chunk=$SEED_CHUNK"
    python deforestation_study/run_wor.py $SEED_CHUNK
elif [ $SCRIPT_IDX -eq 1 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_bernoulli.py seed_chunk=$SEED_CHUNK"
    python deforestation_study/run_bernoulli.py $SEED_CHUNK
elif [ $SCRIPT_IDX -eq 2 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_cross_ppi.py seed_chunk=$SEED_CHUNK"
    python deforestation_study/run_cross_ppi.py $SEED_CHUNK
else
    echo "Task $SLURM_ARRAY_TASK_ID: run_active_inference.py seed_chunk=$SEED_CHUNK"
    python deforestation_study/run_active_inference.py $SEED_CHUNK
fi
