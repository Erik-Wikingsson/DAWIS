#!/bin/bash
# SLURM job-mail options from MAIL_TYPE / MAIL_USER (environment or .env).
#
# Sourced, not executed, after scripts/repo_root.sh. Builds SBATCH_MAIL_ARGS:
#     source "$REPO_ROOT/scripts/sbatch_opts.sh"
#     sbatch "${SBATCH_MAIL_ARGS[@]}" -t "$t" --job-name="$n" job_script.bash
# `#SBATCH` lines cannot read variables, so mail has to go on the command line;
# submit single job scripts with scripts/submit.sh. Default: no mail.

source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

MAIL_TYPE="${MAIL_TYPE:-NONE}"
MAIL_USER="${MAIL_USER:-}"

MAIL_TYPE="$(echo "$MAIL_TYPE" | tr '[:lower:]' '[:upper:]')"

# The values --mail-type accepts.
_sbatch_opts_valid="NONE BEGIN END FAIL REQUEUE ALL INVALID_DEPEND STAGE_OUT
TIME_LIMIT TIME_LIMIT_90 TIME_LIMIT_80 TIME_LIMIT_50 ARRAY_TASKS"
_sbatch_opts_ifs="$IFS"; IFS=','
for _sbatch_opts_tok in $MAIL_TYPE; do
    if [[ " $(echo $_sbatch_opts_valid) " != *" $_sbatch_opts_tok "* ]]; then
        IFS="$_sbatch_opts_ifs"
        echo "ERROR: invalid MAIL_TYPE value: $_sbatch_opts_tok" >&2
        echo "  Valid: $(echo $_sbatch_opts_valid), or a comma-separated list." >&2
        echo "  Set MAIL_TYPE in your environment, or as MAIL_TYPE= in the repo .env." >&2
        exit 1
    fi
done
IFS="$_sbatch_opts_ifs"
unset _sbatch_opts_valid _sbatch_opts_ifs _sbatch_opts_tok

# Anything but NONE needs an address.
if [[ "$MAIL_TYPE" == "NONE" ]]; then
    SBATCH_MAIL_ARGS=(--mail-type=NONE)
elif [[ -z "$MAIL_USER" ]]; then
    echo "ERROR: MAIL_TYPE=$MAIL_TYPE but MAIL_USER is empty." >&2
    echo "  Set MAIL_USER= in the repo .env, or MAIL_TYPE=NONE for no mail." >&2
    exit 1
else
    SBATCH_MAIL_ARGS=(--mail-type="$MAIL_TYPE" --mail-user="$MAIL_USER")
fi

export MAIL_TYPE MAIL_USER   # SBATCH_MAIL_ARGS is an array: source, don't export
