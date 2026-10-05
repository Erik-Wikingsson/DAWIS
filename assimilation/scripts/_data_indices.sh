#!/bin/bash
# DATA_INDICES="0 10" overrides a launcher's trajectory list (environment only).
# Sourced by the launchers; call `apply_data_indices` right before the
# trajectory loop, since `--sweep false` may reassign data_indices earlier.
# An empty or non-numeric value is an error.

apply_data_indices () {
    [[ -n "${DATA_INDICES+set}" ]] || return 0

    local -a _wanted
    read -r -a _wanted <<< "${DATA_INDICES}"
    if [[ ${#_wanted[@]} -eq 0 ]]; then
        echo "ERROR: DATA_INDICES is set but empty -- no jobs would be submitted." >&2
        echo "  Unset it to use the launcher's own data_indices, or give it" >&2
        echo "  whitespace-separated trajectory numbers, e.g. DATA_INDICES=\"0 10\"." >&2
        exit 1
    fi

    local _i
    for _i in "${_wanted[@]}"; do
        if [[ ! "$_i" =~ ^[0-9]+$ ]]; then
            echo "ERROR: DATA_INDICES must be whitespace-separated non-negative" \
                 "integers; got '${_i}' in '${DATA_INDICES}'." >&2
            exit 1
        fi
    done

    data_indices=("${_wanted[@]}")
    echo "DATA_INDICES override: data_indices=(${data_indices[*]})"
}
