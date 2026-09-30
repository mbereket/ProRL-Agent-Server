#!/usr/bin/env bash
# HF -> Megatron torch_dist for the full fine-tune base (raw mode). LoRA runs skip
# this: Megatron-Bridge builds the frozen base straight from the HF checkpoint.
# Runs Miles' converter inside the Miles image. Needs MODEL_ARGS_FILE, HF_CHECKPOINT,
# TORCH_DIST_DIR, HF_HOME, MILES_RUNTIME_DIR (launch.sh exports them).
set -euo pipefail
: "${MODEL_ARGS_FILE:?}" "${HF_CHECKPOINT:?}" "${TORCH_DIST_DIR:?}" "${MILES_RUNTIME_DIR:?}"
# shellcheck disable=SC1090
source "${MODEL_ARGS_FILE}"
hf="${HF_CHECKPOINT}"
case "${hf}" in
    /*) ;;
    *) hf="$(HF_HUB_OFFLINE=1 "${POLAR_PYTHON:?}" -c 'import sys; from huggingface_hub import snapshot_download as d; print(d(sys.argv[1]))' "${hf}")" ;;
esac
tmp="${TORCH_DIST_DIR}.partial.$$"
mkdir -p "$(dirname "${TORCH_DIST_DIR}")"
HF_HUB_OFFLINE=1 bash "${MILES_RUNTIME_DIR}/mrun" -- bash -c 'cd /root/miles && exec torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py "$@"' _ \
    "${MODEL_ARGS[@]}" --hf-checkpoint "${hf}" --save "${tmp}"
mv "${tmp}" "${TORCH_DIST_DIR}"
echo "converted ${hf} -> ${TORCH_DIST_DIR}"
