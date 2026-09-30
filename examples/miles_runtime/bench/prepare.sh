#!/usr/bin/env bash
# Stage the benchmark inputs under $MILES_STACK_ROOT (idempotent; run inside the runtime:
# `mrun bash bench/prepare.sh`). Model: Qwen3.5-9B (HF layout, bridge mode loads it directly).
# Datasets: dapo-math-17k (train) and aime-2024 (eval), the Miles Qwen3.5 recipe's data.
set -euo pipefail
: "${MILES_STACK_ROOT:?}"
export HF_HOME="${HF_HOME:-${MILES_STACK_ROOT}/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=0
MODELS="${MILES_STACK_ROOT}/models"; DATA="${MILES_STACK_ROOT}/datasets"
mkdir -p "${MODELS}" "${DATA}"

stage_model() {  # name hf_id [local snapshot to copy from]
    local name="$1" repo="$2" src="${3:-}" dst="${MODELS}/$1"
    [ -f "${dst}/config.json" ] && { echo "model ${name}: present"; return; }
    if [ -n "${src}" ] && [ -f "${src}/config.json" ]; then
        echo "model ${name}: copying ${src}"
        cp -rL "${src}" "${dst}.partial.$$"
    else
        echo "model ${name}: downloading ${repo}"
        python3 -c "from huggingface_hub import snapshot_download as d; d('${repo}', local_dir='${dst}.partial.$$')"
    fi
    mv "${dst}.partial.$$" "${dst}"
}

stage_dataset() {  # name hf_dataset_id
    local dst="${DATA}/$1"
    [ -n "$(ls "${dst}"/*.jsonl 2>/dev/null)" ] && { echo "dataset $1: present"; return; }
    python3 -c "from huggingface_hub import snapshot_download as d; d('$2', repo_type='dataset', local_dir='${dst}')"
}

QWEN35_9B_SRC="${QWEN35_9B_SRC:-}"
if [ -z "${QWEN35_9B_SRC}" ]; then
    for c in "${MILES_STACK_ROOT}"/../../prorl-harbor/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/*; do
        [ -f "${c}/config.json" ] && QWEN35_9B_SRC="${c}"
    done
fi
stage_model Qwen3.5-9B Qwen/Qwen3.5-9B "${QWEN35_9B_SRC}"
stage_dataset dapo-math-17k zhuzilin/dapo-math-17k
stage_dataset aime-2024 zhuzilin/aime-2024
ls -la "${MODELS}" "${DATA}"/*
du -sh "${MODELS}/Qwen3.5-9B"
