#!/bin/bash
# Generate only the SQG test split, interactively (11 trajectories, ~minutes).
# See data/SQG/generate_splits.sh for the full set of options.
bash "$(dirname "${BASH_SOURCE[0]}")/generate_splits.sh" --splits test --jobs 11 "$@"
