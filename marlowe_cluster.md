# Marlowe Cluster Reference

## Account & Access
- SSH: `ssh maechler@login.marlowe.stanford.edu` (lands on login node e.g. `n24`)
- SLURM account: `marlowe-m000127`
- Partition: `preempt` (max 12 hours, jobs can be preempted)
- Home dir: `/users/maechler` (small quota)
- Scratch dir: `/scratch/m000127/maechler/` (large storage, use for envs and big data)

## Conda Environment
- Miniconda3 location: `/users/maechler/miniconda3`
- Project env: `/scratch/m000127/maechler/faq_env` (Python 3.12, torch, numpy, pandas, scipy, tqdm)
- Activate in scripts:
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate /scratch/m000127/maechler/faq_env
  ```

## SBATCH Script Rules
- `#SBATCH` must have NO space between `#` and `SBATCH` — `# SBATCH` is ignored
- `#SBATCH` lines must start at column 0 — no leading spaces or indentation
- No trailing whitespace on `#SBATCH` lines (can cause parsing issues)
- Copy-paste from terminals/browsers often introduces spaces — use `cat -A file.sh | head` to check

## SBATCH Template
```bash
#!/bin/sh
#SBATCH --job-name=myjob
#SBATCH -p preempt
#SBATCH --nodes=1
#SBATCH -A marlowe-m000127
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --array=0-19

module load slurm
module load nvhpc
module load cudnn/cuda12/9.3.0.75

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/m000127/maechler/faq_env

cd ~/efficiently-evaluating-llms

# your commands here
```

## Job Array Parallelization
- `--array=0-N` launches N+1 jobs, each gets a unique `SLURM_ARRAY_TASK_ID`
- Use integer division/modulo to map task ID to parameter combos:
  ```bash
  dataset=$((SLURM_ARRAY_TASK_ID / 10))
  remainder=$((SLURM_ARRAY_TASK_ID % 10))
  gamma=$((remainder / 2))
  ms=$((remainder % 2))
  ```

## Useful Commands
- Submit: `sbatch script.sh`
- Monitor: `squeue -u $USER`
- Cancel all: `scancel -u $USER`
- Cancel one: `scancel <job_id>`
- Check account: `sacctmgr show associations user=$USER format=account%40,partition%20`
- Quick test: `sbatch -A marlowe-m000127 -p preempt --wrap="echo test"`

## Repo Setup on Cluster
- Repo: `~/efficiently-evaluating-llms`
- Data must be unzipped into separate folders:
  ```bash
  cd data/processed
  unzip mmlu-pro.zip -d mmlu-pro/
  unzip bbh+gpqa+ifeval+math+musr.zip -d bbh+gpqa+ifeval+math+musr/
  ```
- Factor models: `factor_models/val/` and `factor_models/final/`
- Log dirs: `mkdir -p logs/val logs/final`
