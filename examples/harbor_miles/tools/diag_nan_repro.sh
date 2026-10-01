#!/usr/bin/env bash
# DIAG: SGLang near-full-context logprob repro (1 GPU) inside the Miles SIF (same image/patches as the training runs).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
MILES_RUNTIME_DIR="${MILES_RUNTIME_DIR:-${SCOMPOSE_PKGS:-}/miles_runtime}"
MODEL="${HM_SHARED}/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
mkdir -p "${HM_ROOT}/runs/diag-nanrepro"
bash "${MILES_RUNTIME_DIR}/mrun" -- python3 "${HERE}/diag_nan_repro.py" "${MODEL}" 4096 2>&1 | tee "${HM_ROOT}/runs/diag-nanrepro/repro-${SLURM_JOB_ID}.txt"
