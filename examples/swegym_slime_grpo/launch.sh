#!/usr/bin/env bash
# Train on SWE-Gym with Polar + Slime, from one run config.
#
#   bash examples/swegym_slime_grpo/launch.sh configs/<run>.yaml [--dry-run]
#
# The config (see configs/ and README.md) holds the model, GPU layout, rollout
# and training hyperparameters. This script turns it into environment
# variables and hands off to internal/pipeline.sh, which sets up the
# environment (idempotent), prepares data, images and the checkpoint, and
# starts training. --dry-run prints the resolved settings and stops.
#
# Multi-node under slurm: slurm_launch.sh --config configs/<run>.yaml
# Machine-side settings (WORKROOT, ports, WANDB_API_KEY, ...) stay environment
# variables; see README.md.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RUN_CONFIG=""; DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) RUN_CONFIG="${arg}" ;;
    esac
done
[ -n "${RUN_CONFIG}" ] || { echo "usage: launch.sh <run-config.yaml> [--dry-run]" >&2; exit 2; }
[ -f "${RUN_CONFIG}" ] || { echo "ERROR: run config not found: ${RUN_CONFIG}" >&2; exit 2; }

# shellcheck source=./internal/setup/common.sh
source "${SCRIPT_DIR}/internal/setup/common.sh"
cfg_env="$(config_python "${SCRIPT_DIR}/internal/config_to_env.py" "${RUN_CONFIG}")" || exit 1
eval "${cfg_env}"
export RUN_ID="${RUN_ID:-${RUN_NAME}}"
if [ -n "${SLURM_JOB_NUM_NODES:-}" ] && [ "${SLURM_JOB_NUM_NODES}" != "${NUM_NODES}" ]; then
    die "config cluster.num_nodes=${NUM_NODES} but the slurm allocation has ${SLURM_JOB_NUM_NODES} nodes"
fi
export RUN_CONFIG_PATH="$(cd -- "$(dirname -- "${RUN_CONFIG}")" && pwd)/$(basename -- "${RUN_CONFIG}")"

cat <<INFO
Run:       ${RUN_NAME} (RUN_ID ${RUN_ID})
Config:    ${RUN_CONFIG_PATH}
Model:     ${HF_CHECKPOINT} (${MODEL_ARGS_FILE})
Layout:    ${NUM_NODES} node(s); trainer ${ACTOR_NUM_GPUS} GPUs TP${TP_SIZE} x CP${CONTEXT_PARALLEL_SIZE}; every other GPU serves an engine
Rollout:   ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} samples, ${NUM_EPOCH} epoch(s); prompt/response caps ${ROLLOUT_MAX_PROMPT_LEN}/${ROLLOUT_MAX_RESPONSE_LEN}, sglang ctx ${SGLANG_CONTEXT_LENGTH}
Training:  per-trace cap $((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE)) tokens (max_tokens_per_gpu x CP); lr ${LR}; KL=${USE_KL_LOSS} std_norm=${GRPO_STD_NORMALIZATION}; save every ${SAVE_INTERVAL}
INFO
if [ "${DRY_RUN}" = 1 ]; then
    echo "--- environment for internal/pipeline.sh ---"
    echo "${cfg_env}" | sed 's/^export //' | sort
    exit 0
fi
exec bash "${SCRIPT_DIR}/internal/pipeline.sh"
