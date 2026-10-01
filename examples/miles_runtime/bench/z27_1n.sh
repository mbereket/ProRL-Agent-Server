#!/usr/bin/env bash
# 27B 1-node trainer layouts with MTP off (synthetic fixed-length); then print each arm's GPU/node args for verification.
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
bash "${MR}/bench/run.sh" z27-nomtp arms-27b-1n-nomtp.txt bench/patches:patches/no-mtp
for a in "${MILES_STACK_ROOT}"/bench/z27-nomtp/n1-*/args.txt; do
    echo "$(basename "$(dirname "${a}")"): $(grep -A1 -E -- '^--(actor-num-gpus-per-node|actor-num-nodes|tensor-model-parallel-size|context-parallel-size)$' "${a}" | grep -v -- '^--$' | paste -sd' ' -)"
done
