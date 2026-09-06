#!/usr/bin/env bash
# Convert an HF checkpoint to Megatron torch_dist format for Slime.
# Environment (from launch.sh): HF_CHECKPOINT, TORCH_DIST_DIR, MODEL_ARGS_FILE,
# SLIME_DIR, MEGATRON_DIR, PYTHON_BIN, PROJECT_ROOT. Qwen3.5 checkpoints are VLMs;
# weight loading goes through slime_plugins.mbridge.qwen3_5 (text_config-aware).
set -euo pipefail
: "${HF_CHECKPOINT:?}" "${TORCH_DIST_DIR:?}" "${MODEL_ARGS_FILE:?}" "${SLIME_DIR:?}" "${MEGATRON_DIR:?}" "${PYTHON_BIN:?}" "${PROJECT_ROOT:?}"
# shellcheck disable=SC1090
source "${MODEL_ARGS_FILE}"
mkdir -p "${TORCH_DIST_DIR}"
echo "converting ${HF_CHECKPOINT} -> ${TORCH_DIST_DIR}"
CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
"$(dirname "${PYTHON_BIN}")/torchrun" --nproc_per_node 1 "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --save "${TORCH_DIST_DIR}" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion
echo "done: ${TORCH_DIST_DIR}"
