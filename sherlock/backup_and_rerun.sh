#!/bin/bash
# Backup PEW logs on cluster (only study not pulled locally yet),
# then delete old CSVs so checkpointing doesn't skip.
#
# Galaxy, AlphaFold, and WOR baselines were already backed up locally.
# Run this ONCE on the cluster after git pull, before submitting jobs.

set -e

cd /home/groups/candes/maechler/efficient-evaluation-experiments

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Pew study (not transferred locally — backup needed here) ---
if [ -d pew_study/logs ] && ls pew_study/logs/bernoulli_sl=*.csv 1>/dev/null 2>&1; then
    mkdir -p "pew_study/logs_backup_pre_fix"
    cp pew_study/logs/bernoulli_sl=*.csv "pew_study/logs_backup_pre_fix/"
    echo "Backed up pew bernoulli CSVs"
    rm -f pew_study/logs/bernoulli_sl=*.csv
    echo "Deleted old pew bernoulli CSVs"
else
    echo "No pew bernoulli CSVs to backup (already cleaned by git pull)"
fi

echo ""
echo "=== Ready to submit jobs: ==="
echo "  sbatch sherlock/galaxy_study/submit_rerun.sh"
echo "  sbatch sherlock/alphafold_study/submit_rerun.sh"
echo "  sbatch sherlock/alphafold_study/submit_oddsratio.sh"
echo "  sbatch sherlock/pew_study/submit_rerun.sh"
echo "  sbatch sherlock/wor_study/submit_wor_baselines.sh"
