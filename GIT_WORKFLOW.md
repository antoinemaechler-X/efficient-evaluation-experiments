# Git Workflow

## Repo
- **GitHub:** https://github.com/antoinemaechler-X/efficient-evaluation-experiments.git
- **Upstream (original):** https://github.com/skbwu/efficiently-evaluating-llms.git

## Machines
| Machine | Path | Remote `origin` | Remote `upstream` |
|---------|------|-----------------|-------------------|
| Local (Mac) | `~/efficiently-evaluating-llms` | your GitHub repo | skbwu's repo |
| Cluster (Marlowe) | `~/efficiently-evaluating-llms` | your GitHub repo | skbwu's repo |

## Day-to-day workflow

### Code changes (local, via Claude Code)
```bash
# After Claude makes edits locally:
git add <files>
git commit -m "description"
git push origin main
```

### Pull on the cluster
```bash
cd ~/efficiently-evaluating-llms
git pull origin main
```

### Experiment results (cluster → GitHub)
```bash
# On cluster, after experiments produce new files:
git add <files>
git commit -m "description"
git push origin main
```

### Pull results locally
```bash
cd ~/efficiently-evaluating-llms
git pull origin main
```

## Setup (already done)
- Both machines have `main` tracking `origin/main` (`git branch --set-upstream-to=origin/main main`), so plain `git pull` / `git push` works.

## Rules
- Always commit or stash before pulling to avoid conflicts.
- Large data files (`data/raw.zip`, `data/processed/`, `acs_study/data/`) are in `.gitignore` — never tracked.
- Experiment logs and figures are tracked (so results are preserved).

## Conda environments
| Machine | Env | Activate |
|---------|-----|----------|
| Local | `acs_env` | `conda activate acs_env` |
| Cluster | `faq_env` | `conda activate /scratch/m000127/maechler/faq_env` |
