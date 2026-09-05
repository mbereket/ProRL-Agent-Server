#!/usr/bin/env bash
# GRPO training on a directory of Harbor tasks with Polar + Slime, from one run config.
#
#   bash examples/harbor_slime_grpo/launch.sh configs/<run>.yaml [--dry-run]
#
# The config (see configs/ and README.md) holds the task directory and subset,
# the agent harness, the model, GPU layout, rollout and training
# hyperparameters. This script turns it into environment variables and hands
# off to internal/pipeline.sh, which sets up the environment (idempotent),
# prepares prompts, images, the harness and the checkpoint, renders the Polar
# templates and starts training. --dry-run resolves the config, builds the
# prompt list and renders the templates without touching GPUs, images or
# checkpoints.
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
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
export RUN_CONFIG_PATH="$(cd -- "$(dirname -- "${RUN_CONFIG}")" && pwd)/$(basename -- "${RUN_CONFIG}")"
export DRY_RUN
if [ -n "${SLURM_JOB_NUM_NODES:-}" ] && [ "${SLURM_JOB_NUM_NODES}" != "${NUM_NODES}" ]; then
    die "config cluster.num_nodes=${NUM_NODES} but the slurm allocation has ${SLURM_JOB_NUM_NODES} nodes"
fi
if [ ! -d "${TASKS_DIR}" ] && [ -z "${TASKS_DATASET}" ]; then
    die "tasks.dir does not exist: ${TASKS_DIR} (set tasks.dataset to have datasets/<name>.py create it)"
fi

cat <<INFO
Run:       ${RUN_NAME} (RUN_ID ${RUN_ID})
Config:    ${RUN_CONFIG_PATH}
Tasks:     ${TASKS_DIR}${TASKS_N:+ (n=${TASKS_N}, seed ${TASKS_SEED})}
Harness:   ${HARNESS}
Model:     ${HF_CHECKPOINT} (${MODEL_ARGS_FILE})
Layout:    ${NUM_NODES} node(s); $([ "${COLOCATE:-0}" = 1 ] && echo "colocated: all ${ACTOR_NUM_GPUS} GPUs train (TP${TP_SIZE} x CP${CONTEXT_PARALLEL_SIZE}) and serve engines" || echo "trainer ${ACTOR_NUM_GPUS} GPUs TP${TP_SIZE} x CP${CONTEXT_PARALLEL_SIZE}; every other GPU serves an engine"); sandboxes on ${SANDBOX_NODES} node(s)
Rollout:   ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} samples, ${NUM_EPOCH} epoch(s); prompt/response caps ${ROLLOUT_MAX_PROMPT_LEN}/${ROLLOUT_MAX_RESPONSE_LEN}, sglang ctx ${SGLANG_CONTEXT_LENGTH}
Training:  ${TRAIN_SCRIPT}; per-trace cap $((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE)) tokens (max_tokens_per_gpu x CP); lr ${LR}; KL=${USE_KL_LOSS} std_norm=${GRPO_STD_NORMALIZATION} scope=${GROUP_ID_SCOPE}; save every ${SAVE_INTERVAL}
INFO
if [ "${DRY_RUN}" = 1 ]; then
    echo "--- environment for internal/pipeline.sh ---"
    echo "${cfg_env}" | sed 's/^export //' | sort
fi
exec bash "${SCRIPT_DIR}/internal/pipeline.sh"
