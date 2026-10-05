#!/bin/bash
#SBATCH -J Gen_DATA
#SBATCH -t 03-00:00:00
#SBATCH --partition=berzelius-cpu
#SBATCH -c 32
# SLURM wrapper around data/SQG/generate_splits.sh, one process per CPU:
#   scripts/submit.sh data/SQG/gen_data.sh [generate_splits.sh options]
# The partition above is Berzelius-specific; change it for your cluster.

# $REPO_ROOT is propagated by sbatch from scripts/submit.sh; a submitted script
# runs from a spool copy, so a path relative to it cannot find the repo.
_conda_env_sh="${REPO_ROOT:-$(dirname "${BASH_SOURCE[0]}")/../..}/scripts/conda_env.sh"
if [[ ! -f "$_conda_env_sh" ]]; then
    echo "ERROR: no scripts/conda_env.sh at: $_conda_env_sh" >&2
    echo "  \$REPO_ROOT is unset. Submit with scripts/submit.sh, which sets it." >&2
    exit 1
fi
source "$_conda_env_sh"
unset _conda_env_sh

bash "$REPO_ROOT/data/SQG/generate_splits.sh" --jobs "${SLURM_CPUS_PER_TASK:-1}" "$@"
