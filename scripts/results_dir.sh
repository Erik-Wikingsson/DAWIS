#!/bin/bash
# Resolve the results directory (RESULTS_ROOT) and export RESULTS_DIR.
#
# Sourced, not executed. data/paths.py:results_dir is the Python equivalent.

source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

# Group-writable, matching RESULTS_DIR_MODE / RESULTS_FILE_MODE in data/paths.py.
RESULTS_DIR_MODE=2775
RESULTS_FILE_MODE=664

if [[ -z "${RESULTS_ROOT:-}" ]]; then
    echo "ERROR: RESULTS_ROOT is not set." >&2
    echo "  Set it in your environment, or as RESULTS_ROOT= in the repo .env." >&2
    exit 1
fi

RESULTS_DIR="$RESULTS_ROOT"

if ! mkdir -p "$RESULTS_DIR"; then
    echo "ERROR: could not create the results directory: $RESULTS_DIR" >&2
    exit 1
fi
chmod "$RESULTS_DIR_MODE" "$RESULTS_DIR" 2>/dev/null || true

if [[ ! -w "$RESULTS_DIR" ]]; then
    echo "ERROR: the results directory is not writable: $RESULTS_DIR" >&2
    exit 1
fi

export RESULTS_ROOT RESULTS_DIR RESULTS_DIR_MODE RESULTS_FILE_MODE

# Make this user's recent results group-readable. Only touches files the
# current user owns that are younger than $1 days (default 1).
fix_results_permissions() {
    local since="${1:-1}"
    echo "Fixing permissions in $RESULTS_DIR ..."
    find "$RESULTS_DIR" -maxdepth 1 -user "$(id -u)" -type f -mtime "-${since}" \
        -exec chmod "$RESULTS_FILE_MODE" {} + 2>/dev/null || true
    chmod "$RESULTS_DIR_MODE" "$RESULTS_DIR" 2>/dev/null || true
    echo "Done!"
}
