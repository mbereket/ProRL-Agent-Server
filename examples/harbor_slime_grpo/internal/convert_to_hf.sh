#!/usr/bin/env bash
# Export a training checkpoint (Megatron torch_dist) as an HF-format model directory,
# servable by vLLM / SGLang for evaluation.
#
#   convert_to_hf.sh <run-name> <iteration> <output-dir>
#
# Reads ${WORKROOT}/ckpt/harbor_slime_grpo/<run-name>/iter_<iteration padded to 7> and
# writes <output-dir> (config, tokenizer and any weights the trainer does not hold,
# e.g. the vision tower of a Qwen3.5 checkpoint, are copied from the base HF snapshot).
# Needs the environment launch.sh built: the venv, the slime + Megatron checkouts and
# the HF snapshot under ${WORKROOT}. One GPU, a few minutes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${HERE}/../../.." && pwd)}"
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
RUN_NAME="${1:?usage: convert_to_hf.sh <run-name> <iteration> <output-dir>}"
ITER="${2:?usage: convert_to_hf.sh <run-name> <iteration> <output-dir>}"
OUT_DIR="${3:?usage: convert_to_hf.sh <run-name> <iteration> <output-dir>}"

SAVE_DIR="${WORKROOT}/ckpt/harbor_slime_grpo/${RUN_NAME}"
ITER_DIR="$(printf '%s/iter_%07d' "${SAVE_DIR}" "${ITER}")"
[ -d "${ITER_DIR}" ] || { echo "ERROR: no checkpoint at ${ITER_DIR} (have: $(ls "${SAVE_DIR}" 2>/dev/null | tr '\n' ' '))" >&2; exit 1; }
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "ERROR: venv not found at ${PYTHON_BIN}; run launch.sh --setup-only first" >&2; exit 1; }
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REF="$(sed -n 's/^slime = { git = "[^"]*", rev = "\([0-9a-f]*\)".*/\1/p' "${HERE}/setup/stack/pyproject.toml")"
MEGATRON_DIR="${MEGATRON_DIR:-${WORKROOT}/Megatron-LM-slime-${SLIME_REF:0:12}}"
[ -d "${MEGATRON_DIR}/megatron" ] || { echo "ERROR: Megatron checkout not found at ${MEGATRON_DIR}" >&2; exit 1; }
export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}" HF_HUB_OFFLINE=1

# The base model the run trained from (HF_CHECKPOINT in the run's rendered env.sh).
RUN_ENV="${WORKROOT}/harbor_slime_grpo/${RUN_NAME}/env.sh"
[ -f "${RUN_ENV}" ] || { echo "ERROR: run env not found at ${RUN_ENV}" >&2; exit 1; }
HF_CHECKPOINT="$(bash -c "source '${RUN_ENV}' && printf %s \"\${HF_CHECKPOINT}\"")"
[ -n "${HF_CHECKPOINT}" ] || { echo "ERROR: HF_CHECKPOINT not set in ${RUN_ENV}" >&2; exit 1; }
ORIGIN_HF="$("${PYTHON_BIN}" -c "from huggingface_hub import snapshot_download; print(snapshot_download('${HF_CHECKPOINT}'))")"
echo "converting ${ITER_DIR} -> ${OUT_DIR} (base ${HF_CHECKPOINT} at ${ORIGIN_HF})"
mkdir -p "$(dirname "${OUT_DIR}")"
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" "${PYTHON_BIN}" "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" \
    --input-dir "${ITER_DIR}" --output-dir "${OUT_DIR}" \
    --origin-hf-dir "${ORIGIN_HF}" --add-missing-from-origin-hf --force
echo "HF checkpoint ready: ${OUT_DIR}"
ls "${OUT_DIR}"
