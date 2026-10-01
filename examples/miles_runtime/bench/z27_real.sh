#!/usr/bin/env bash
# 27B trainer layouts on real long traces, MTP off; then print each arm's GPU/TP/CP args for verification.
#   z27_real.sh [ARMS_FILE]   (default arms-27b-real-dfw.txt)
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
bash "${MR}/bench/run.sh" z27-real "${1:-arms-27b-real-dfw.txt}" bench/patches:patches/no-mtp
for a in "${MILES_STACK_ROOT}"/bench/z27-real/r-*/args.txt; do
    echo "$(basename "$(dirname "${a}")"): $(grep -A1 -E -- '^--(actor-num-gpus-per-node|actor-num-nodes|tensor-model-parallel-size|context-parallel-size)$' "${a}" | grep -v -- '^--$' | paste -sd' ' -)"
done
