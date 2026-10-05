#!/bin/bash
# sbatch, with this checkout's shared mail options spliced in.
#
#     scripts/submit.sh data/SQG/gen_data.sh
#     scripts/submit.sh -t 04:00:00 unconditional_generation/scripts/SQG/train_64.sh
#
# Use it for job scripts you submit by hand. The run_all_*.bash launchers
# already splice the same options in themselves, so submit *through* them as
# before -- this is not a wrapper for those.
#
# Everything after the command name is passed to sbatch untouched, and lands
# after the mail options, so an explicit --mail-type/--mail-user here wins for
# that one submission.
#
# Plain `sbatch job_script.bash` still works; it just sends no mail, because
# a #SBATCH directive cannot read MAIL_USER. See scripts/sbatch_opts.sh.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/repo_root.sh"
source "$REPO_ROOT/scripts/sbatch_opts.sh"

if [[ $# -eq 0 ]]; then
    echo "usage: scripts/submit.sh [sbatch options] <job script> [script args]" >&2
    exit 2
fi

exec sbatch "${SBATCH_MAIL_ARGS[@]}" "$@"
