#!/bin/bash
#SBATCH --job-name=af_or
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
##SBATCH -A candes
#SBATCH --time=12:00:00
#SBATCH --output=alphafold_study/logs/slurm_af_or_%A_%a.out
#SBATCH --error=alphafold_study/logs/slurm_af_or_%A_%a.err
#SBATCH --array=0-5

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 6 jobs: 2 scripts × 3 seed chunks (NEW odds ratio estimators)
# 0-2: run_bernoulli_oddsratio.py (Zrnic's method)
# 3-5: run_wor_oddsratio.py (our method)

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    echo "Task $SLURM_ARRAY_TASK_ID: run_bernoulli_oddsratio.py seed_chunk=$SEED_CHUNK"
    python alphafold_study/run_bernoulli_oddsratio.py $SEED_CHUNK
else
    echo "Task $SLURM_ARRAY_TASK_ID: run_wor_oddsratio.py seed_chunk=$SEED_CHUNK"
    python alphafold_study/run_wor_oddsratio.py $SEED_CHUNK
fi
