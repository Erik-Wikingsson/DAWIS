#!/bin/bash
# Resolve the repo root, cd there and set PYTHONPATH. Also loads the repo .env.
#
# Sourced, not executed:
#     source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/repo_root.sh"
#
# REPO_ROOT (environment or .env) points at a different checkout; by default
# it is the checkout this file lives in.

source "$(dirname "${BASH_SOURCE[0]}")/load_env.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [[ ! -d "$REPO_ROOT" ]]; then
    echo "ERROR: REPO_ROOT is not a directory: $REPO_ROOT" >&2
    echo "  Set it in your environment, or as REPO_ROOT= in the repo .env." >&2
    exit 1
fi
if [[ ! -f "$REPO_ROOT/assimilation/assimilate.py" ]]; then
    echo "ERROR: REPO_ROOT does not look like this repo: $REPO_ROOT" >&2
    echo "  (no assimilation/assimilate.py beneath it)" >&2
    exit 1
fi

export REPO_ROOT
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
