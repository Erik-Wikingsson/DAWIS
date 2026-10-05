#!/bin/bash
# Generate the SQG train/val/test splits in the layout data/SQG/config.yaml
# expects, under $SQG_ROOT, and compute the normalization statistics.
#
# Usage:
#   bash data/SQG/generate_splits.sh [options]
#
# Options:
#   --variant base|hires   base: 64x64 (default), hires: 256x256
#   --splits "train val test test_long"
#                          which splits to generate (default: all of them;
#                          hires has no test_long)
#   --n_train N            number of training trajectories (default: 2000 / 10)
#   --n_val N              (default: 1)
#   --n_test N             (default: 11; index 0 is used for tuning)
#   --jobs J               parallel processes (default: 1)
#   --seed S               base seed (default: 0); each split and trajectory
#                          gets its own seed derived from it
#   --no_stats             skip compute_data_stats.py
#
# Layout written (base):
#   train/64_3h/sqg_N64_3hrly_steps_100_<i>_<id>.{nc,npy}   101 frames each
#   val/64_3h/sqg_N64_3hrly_steps_100_<i>_<id>.{nc,npy}     101 frames
#   test/64_3h/sqg_N64_3hrly_steps_110_<i>_<id>.{nc,npy}    111 frames
#   test/64_3h_1000step/sqg_N64_3hrly_steps_1000_0_<id>.{nc,npy}   1001 frames
#   train/64_3h/{data,diff}_{mean,std}.pt
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/repo_root.sh"

variant=base
splits=""
n_train=""
n_val=1
n_test=""
jobs=1
seed=0
stats=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) variant="$2"; shift ;;
        --splits)  splits="$2"; shift ;;
        --n_train) n_train="$2"; shift ;;
        --n_val)   n_val="$2"; shift ;;
        --n_test)  n_test="$2"; shift ;;
        --jobs)    jobs="$2"; shift ;;
        --seed)    seed="$2"; shift ;;
        --no_stats) stats=false ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

: "${SQG_ROOT:?set SQG_ROOT in the repo .env (see .env.example)}"

# split -> "<n_traj> <n_times> <directory>"
case "$variant" in
    base)
        N=64
        splits="${splits:-train val test test_long}"
        declare -A spec=([train]="${n_train:-2000} 100 train/64_3h"
                         [val]="$n_val 100 val/64_3h"
                         [test]="${n_test:-11} 110 test/64_3h"
                         [test_long]="1 1000 test/64_3h_1000step") ;;
    hires)
        N=256
        splits="${splits:-train val test}"
        declare -A spec=([train]="${n_train:-10} 1000 train/256_3h"
                         [val]="$n_val 100 val/256_3h"
                         [test]="${n_test:-1} 1000 test/256_3h") ;;
    *) echo "unknown variant: $variant (base or hires)" >&2; exit 2 ;;
esac
declare -A seed_offset=([train]=0 [val]=1000000 [test]=2000000 [test_long]=3000000)

# Run `count` trajectories starting at `first`, split over $jobs processes.
generate() {
    local split="$1" count="$2" n_times="$3" out="$4"
    local per=$(( (count + jobs - 1) / jobs )) first=0 pids=()
    while (( first < count )); do
        local n=$(( count - first < per ? count - first : per ))
        OMP_NUM_THREADS=1 python3 data/SQG/sqg_nature_run.py \
            --N "$N" --hrs 3 --n_times "$n_times" --n_traj "$n" \
            --start_index "$first" --seed $(( seed + ${seed_offset[$split]} )) \
            --data_path "$out" > "$out/generate_${first}.log" 2>&1 &
        pids+=($!)
        first=$(( first + n ))
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if (( failed )); then
        echo "ERROR: generating $split failed; see $out/generate_*.log" >&2
        exit 1
    fi
    rm -f "$out"/generate_*.log
}

for split in $splits; do
    [[ -n "${spec[$split]:-}" ]] || { echo "unknown split: $split" >&2; exit 2; }
    read -r count n_times dir <<< "${spec[$split]}"
    out="$SQG_ROOT/$dir"
    mkdir -p "$out"
    if compgen -G "$out/sqg_N${N}_3hrly_*.npy" > /dev/null; then
        echo "ERROR: $out already holds trajectories; move them away first." >&2
        exit 1
    fi
    echo "Generating $split: $count trajectories of $n_times steps into $out ($jobs jobs)"
    generate "$split" "$count" "$n_times" "$out"
done

if $stats && [[ " $splits " == *" train "* ]]; then
    echo "Computing normalization statistics"
    python3 data/compute_data_stats.py --dataset SQG --variant "$variant" \
        --split train --in_place
fi
echo "Done."
