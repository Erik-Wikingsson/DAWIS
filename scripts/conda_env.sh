#!/bin/bash
# Activate this project's conda environment in a job script.
#
# Sourced, not executed, and before `set -u` (NSC's mamba wrapper reads unset
# variables). CONDA_ENV and MAMBA_MODULE come from the environment or the repo
# .env. MAMBA_MODULE is the environment module providing mamba on Berzelius;
# set it to an empty string on a machine without environment modules.

source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

CONDA_ENV="${CONDA_ENV:-BZ31}"
MAMBA_MODULE="${MAMBA_MODULE-Mambaforge/23.3.1-1-hpc1-bdist}"

if [[ -n "$MAMBA_MODULE" ]]; then
    module load "$MAMBA_MODULE"
fi
if type mamba >/dev/null 2>&1; then
    mamba activate "$CONDA_ENV"
else
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

# The exit status of `mamba activate` does not survive the NSC wrapper, so
# check the variable it sets instead.
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" \
   && "${CONDA_DEFAULT_ENV:-}" != "$(basename "$CONDA_ENV")" ]]; then
    echo "ERROR: could not activate conda environment: $CONDA_ENV" >&2
    echo "  Set CONDA_ENV in your environment, or as CONDA_ENV= in the repo .env." >&2
    conda env list >&2 || true
    exit 1
fi

export CONDA_ENV MAMBA_MODULE
