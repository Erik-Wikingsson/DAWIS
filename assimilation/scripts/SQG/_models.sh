#!/bin/bash
# SQG FMW window models, one per --init_states. Sourced, not executed.
#
#     sqg_fmw_ckpt <init_states> [eta01_channel]
#
# Prints ASSIM_FMW_PATH_SQG_<LOSS>_INIT_<n> if set, else the path under
# MODELS_ROOT, else nothing (assimilate.py then reports what is missing).
# Mirrors FMW_MODELS["SQG"] in assimilation/checkpoints.py: only the released
# window (6) is listed; set the variable above for any other. The subpath is the
# one in the released download (Erik-Wikingsson/dawis-checkpoints); do not rename.

sqg_fmw_ckpt () {
    local init="$1" loss="${2:-eta01_channel}" override sub
    override="ASSIM_FMW_PATH_SQG_${loss^^}_INIT_${init}"
    if [[ -n "${!override:-}" ]]; then
        echo "${!override}"
        return 0
    fi
    case "$loss:$init" in
        eta01_channel:6) sub="eta_channel_init_6-FMW-224-06_28_18-3280" ;;
        *)
            echo "ERROR: no SQG FMW checkpoint for ${loss}, init_states=${init}." >&2
            return 1 ;;
    esac
    [[ -n "${MODELS_ROOT:-}" ]] && echo "${MODELS_ROOT}/DAWIS/models/${sub}/last.ckpt"
}
