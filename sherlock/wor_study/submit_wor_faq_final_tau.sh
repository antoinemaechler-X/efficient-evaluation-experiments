#!/bin/bash
#SBATCH --job-name=wor_final_tau
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# Uncomment and set your account (check with: sacctmgr show associations user=$USER)
##SBATCH -A candes
#SBATCH --time=48:00:00
#SBATCH --output=logs/final/slurm_wor_final_tau_%A_%a.out
#SBATCH --error=logs/final/slurm_wor_final_tau_%A_%a.err
#SBATCH --array=0-5

source /home/groups/gbrice/maechler/Amy_stabl/stabl_env/bin/activate

cd /home/groups/candes/maechler/efficient-evaluation-experiments

# 6 jobs: 2 tau values x 3 seed chunks
# task 0-2: tau=0.25, seed_chunk=0,1,2
# task 3-5: tau=0.5,  seed_chunk=0,1,2
TAUS=(0.25 0.25 0.25 0.5 0.5 0.5)
SEED_CHUNKS=(0 1 2 0 1 2)

TAU=${TAUS[$SLURM_ARRAY_TASK_ID]}
SEED_CHUNK=${SEED_CHUNKS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: tau=$TAU, seed_chunk=$SEED_CHUNK"
python wor_faq_final_tau.py $SEED_CHUNK $TAU
