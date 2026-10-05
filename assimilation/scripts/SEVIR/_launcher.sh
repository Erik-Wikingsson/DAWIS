#!/bin/bash
# Shared preamble for the SEVIR run_all_*.sh launchers. Sourced first (it cd's
# to $REPO_ROOT).
#
# Knobs are passed to the worker through the environment (sbatch --export=ALL),
# so a variable already exported in your shell reaches every job. Each worker
# prints its resolved configuration.

set -euo pipefail

# cwd, $REPO_ROOT and $PYTHONPATH (see scripts/repo_root.sh).
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"

# Job-mail options (see scripts/sbatch_opts.sh).
source "$REPO_ROOT/scripts/sbatch_opts.sh"
mkdir -p slurm_logs

#   DRY_RUN=1 bash assimilation/scripts/SEVIR/run_all_<X>.sh   # print, don't submit
#
# DRY_RUN: supported
# (marker: every sbatch below is behind the DRY_RUN branch in sevir_submit)
DRY_RUN="${DRY_RUN:-0}"

# --sweep true|false sets $SWEEP (see assimilation/scripts/_sweep_arg.sh).
source "$REPO_ROOT/assimilation/scripts/_sweep_arg.sh" "$@"

# NODE_TYPE: thin (A100 40GB) or fat (A100 80GB); overrides the launcher's
# default. The window methods on SEVIR need fat nodes.
NODE_TYPE="${NODE_TYPE:-}"

# Observation interval (see _common.sh). A non-1 interval adds `_ai<k>` to the
# run name, before the trailing `_run_<N>`.
ASSIM_INTERVAL="${ASSIM_INTERVAL:-1}"
export ASSIM_INTERVAL

_ai_tag () {
    local name="$1"
    if [[ "$ASSIM_INTERVAL" == "1" ]]; then
        echo "$name"
    elif [[ "$name" =~ ^(.*)(_run_[0-9]+)$ ]]; then
        echo "${BASH_REMATCH[1]}_ai${ASSIM_INTERVAL}${BASH_REMATCH[2]}"
    else
        echo "${name}_ai${ASSIM_INTERVAL}"
    fi
}

# DATA_INDICES=... overrides a launcher's trajectory list.
source "$REPO_ROOT/assimilation/scripts/_data_indices.sh"

# INIT_STATES_VALUES=... overrides the window depth.
source "$REPO_ROOT/assimilation/scripts/_init_states.sh"

# Windows with an FMW checkpoint; keep in step with `fmw_ckpt_for` in _common.sh.
# Others are allowed when <prefix>_INIT_<n> is set.
_init_states_supported="6"
_init_states_override_prefix="ASSIM_FMW_PATH_SEVIR_ETA01_CHANNEL"

SEVIR_SCRIPTS="assimilation/scripts/SEVIR"
N_JOBS=0

# sevir_submit <job-name> <worker.sh> [sbatch args...] -- [VAR=VALUE ...]
#
# The `--` is required even with no sbatch args.
sevir_submit () {
    local name worker
    name="$(_ai_tag "$1")" worker="$2"
    shift 2
    local sbatch_extra=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        sbatch_extra+=("$1")
        shift
    done
    if [[ "${1:-}" != "--" ]]; then
        echo "sevir_submit: missing '--' between the sbatch args and the" \
             "VAR=VALUE list (job ${name})" >&2
        exit 1
    fi
    shift

    N_JOBS=$((N_JOBS + 1))
    echo "[${N_JOBS}] ${name}"
    if [[ "$DRY_RUN" == "1" ]]; then
        # Print the worker too, so the python argv can be reconstructed.
        printf '        WORKER=%s\n' "$worker"
        printf '        %s\n' "$@"
        return 0
    fi
    # EXP_NAME comes last so the sweep cannot override it.
    env "$@" EXP_NAME="${name}" \
        sbatch "${SBATCH_MAIL_ARGS[@]}" "${sbatch_extra[@]}" \
               --job-name="${name}" "${SEVIR_SCRIPTS}/${worker}"
}

# Fail on an empty sweep array instead of submitting 0 jobs.
require_nonempty () {
    local arr
    for arr in "$@"; do
        local -n _ref="$arr"
        if [[ ${#_ref[@]} -eq 0 ]]; then
            echo "ERROR: sweep array '${arr}' is empty -- no jobs would be" \
                 "submitted." >&2
            exit 1
        fi
        unset -n _ref
    done
}

sevir_summary () {
    echo
    echo "Total ${1} jobs: ${N_JOBS} (dry_run=${DRY_RUN})"
}

# Each launcher declares `experiments`, `data_indices` and (where applicable)
# `sweep` arrays near the top. data_indices: 0 = tuning, 1-10 = evaluation.
