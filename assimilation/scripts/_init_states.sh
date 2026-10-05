#!/bin/bash
# INIT_STATES_VALUES="2 4 6" overrides a launcher's conditioning-window depths
# (--init_states; the window holds init_states + 1 states). Environment only.
# Call `apply_init_states` right before the loop, like apply_data_indices.
# Released checkpoints: window 6 (SQG and SEVIR); a window trained locally is
# used by setting its ASSIM_FMW_PATH_<DATASET>_ETA01_CHANNEL_INIT_<n>. SDA and
# Gibbs need an even window, and --forward_model flowdas needs 6.

# Launcher's own window(s), captured by apply_init_states.
_init_states_default=()

# Windows with FMW checkpoints, set by the launcher; empty skips the check.
_init_states_supported="${_init_states_supported:-}"
# A window outside that list is accepted when <prefix>_INIT_<n> is set.
_init_states_override_prefix="${_init_states_override_prefix:-}"

apply_init_states () {
    # Remember the default so init_states_tag can recognise it.
    _init_states_default=("${init_states_values[@]}")

    [[ -n "${INIT_STATES_VALUES+set}" ]] || return 0

    local -a _wanted
    read -r -a _wanted <<< "${INIT_STATES_VALUES}"
    if [[ ${#_wanted[@]} -eq 0 ]]; then
        echo "ERROR: INIT_STATES_VALUES is set but empty -- no jobs would be submitted." >&2
        echo "  Unset it to use the launcher's own init_states_values, or give it" >&2
        echo "  whitespace-separated window depths, e.g. INIT_STATES_VALUES=\"2 4 6\"." >&2
        exit 1
    fi

    local _w
    for _w in "${_wanted[@]}"; do
        if [[ ! "$_w" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: INIT_STATES_VALUES must be whitespace-separated positive" \
                 "integers; got '${_w}' in '${INIT_STATES_VALUES}'." >&2
            exit 1
        fi
    done

    init_states_values=("${_wanted[@]}")
    echo "INIT_STATES_VALUES override: init_states_values=(${init_states_values[*]})"

    # Refuse windows with no trained checkpoint (override only).
    [[ -n "$_init_states_supported" ]] || return 0
    local _override _where
    for _w in "${_wanted[@]}"; do
        [[ " $_init_states_supported " == *" $_w "* ]] && continue
        _where="its path in the checkpoint table"
        if [[ -n "$_init_states_override_prefix" ]]; then
            _override="${_init_states_override_prefix}_INIT_${_w}"
            [[ -n "${!_override:-}" ]] && continue
            _where="$_override"
        fi
        echo "ERROR: no FMW checkpoint for init_states=${_w}." >&2
        echo "  This launcher has: ${_init_states_supported}." >&2
        echo "  Train one (forecasting/training_scripts/) and set" >&2
        echo "  ${_where}, or drop ${_w} from INIT_STATES_VALUES." >&2
        exit 1
    done
}

# Run-name suffix for a window: "" for the launcher's own default, else _init<N>.
init_states_tag () {
    local _w="$1"
    if [[ ${#_init_states_default[@]} -eq 1 && "$_w" == "${_init_states_default[0]}" ]]; then
        echo ""
    else
        echo "_init${_w}"
    fi
}
