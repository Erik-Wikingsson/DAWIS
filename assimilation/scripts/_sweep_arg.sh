#!/bin/bash
# Parses `--sweep true|false` (or SWEEP=true|false) for the run_all_* launchers.
# Sourced with the launcher's arguments: source .../_sweep_arg.sh "$@"
# Sets and exports $SWEEP (default true = hyperparameter grid; false = the
# tuned configuration used for the paper runs). Unknown arguments are an error.

_sweep_arg_die () {
    echo "ERROR: $*" >&2
    echo "  usage: bash <launcher> [--sweep true|false]" >&2
    echo "    --sweep false  the tuned point behind this method's paper_runs" >&2
    echo "    --sweep true   the hyperparameter grid (the default)" >&2
    exit 1
}

_sweep_arg_check () {
    case "$1" in
        true|false) ;;
        *) _sweep_arg_die "--sweep takes 'true' or 'false', not '$1'." ;;
    esac
}

SWEEP="${SWEEP:-true}"
_sweep_arg_check "$SWEEP"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sweep)
            [[ $# -ge 2 ]] || _sweep_arg_die "--sweep needs a value."
            SWEEP="$2"
            _sweep_arg_check "$SWEEP"
            shift 2
            ;;
        --sweep=*)
            SWEEP="${1#--sweep=}"
            _sweep_arg_check "$SWEEP"
            shift
            ;;
        -h|--help)
            _sweep_arg_die "help requested."
            ;;
        *)
            _sweep_arg_die "unknown argument '$1'."
            ;;
    esac
done

export SWEEP
unset -f _sweep_arg_check
