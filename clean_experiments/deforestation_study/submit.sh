#!/bin/sh
#SBATCH --job-name=deforest
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_deforest_%A_%a.out
#SBATCH --error=results/slurm_deforest_%A_%a.err
#SBATCH --array=0-11

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms/clean_experiments/deforestation_study

# 12 jobs: 4 scripts x 3 seed chunks
# 0-2:  run_wor.py               (GPU)
# 3-5:  run_bernoulli.py          (GPU)
# 6-8:  run_cross_ppi.py          (CPU-only, long runtime)
# 9-11: run_active_inference.py   (GPU)

SCRIPT_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_CHUNK=$((SLURM_ARRAY_TASK_ID % 3))

if [ $SCRIPT_IDX -eq 0 ]; then
    python run_wor.py $SEED_CHUNK
elif [ $SCRIPT_IDX -eq 1 ]; then
    python run_bernoulli.py $SEED_CHUNK
elif [ $SCRIPT_IDX -eq 2 ]; then
    python run_cross_ppi.py $SEED_CHUNK
else
    python run_active_inference.py $SEED_CHUNK
fi
