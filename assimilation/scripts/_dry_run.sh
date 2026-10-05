#!/bin/bash
# DRY_RUN=1 support for the launchers: shadows `sbatch` with a function that
# prints each command (one DRYRUN-SBATCH line per job) instead of submitting.
# Source after scripts/sbatch_opts.sh and before the first submission.

DRY_RUN="${DRY_RUN:-0}"

# Counts submitted (or would-be) jobs in both modes.
DRY_RUN_N_JOBS=0

sbatch () {
    DRY_RUN_N_JOBS=$((DRY_RUN_N_JOBS + 1))
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        # %q per argument keeps arguments containing spaces intact.
        printf 'DRYRUN-SBATCH\t'
        printf '%q ' "$@"
        printf '\n'

        return 0
    fi
    command sbatch "$@"
}

# DRY_RUN: supported
# (Marker for tooling: every sbatch in a script sourcing this file is dry-run safe.)
