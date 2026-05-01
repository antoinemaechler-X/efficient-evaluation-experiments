#!/bin/sh
#SBATCH --job-name=wor_final_tau
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/final/slurm_wor_final_tau_%A_%a.out
#SBATCH --error=logs/final/slurm_wor_final_tau_%A_%a.err
#SBATCH --array=0-5

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

# 6 jobs: 2 tau values x 3 seed chunks
# task 0-2: tau=0.25, seed_chunk=0,1,2
# task 3-5: tau=0.5,  seed_chunk=0,1,2
TAUS=(0.25 0.25 0.25 0.5 0.5 0.5)
SEED_CHUNKS=(0 1 2 0 1 2)

TAU=${TAUS[$SLURM_ARRAY_TASK_ID]}
SEED_CHUNK=${SEED_CHUNKS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: tau=$TAU, seed_chunk=$SEED_CHUNK"
python wor_faq_final_tau.py $SEED_CHUNK $TAU
