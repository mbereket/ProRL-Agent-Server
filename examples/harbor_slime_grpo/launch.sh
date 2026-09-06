#!/usr/bin/env bash
# GRPO on a directory of Harbor tasks with Polar + Slime, from one run config.
#
#   bash launch.sh configs/<run>.yaml                  # run here (single node, or the head of a bare multi-node setup)
#   bash launch.sh configs/<run>.yaml --dry-run        # resolve config, build prompts, render Polar configs; no GPUs
#   bash launch.sh configs/<run>.yaml --setup-only     # environment, images check, harness, checkpoint; no training
#   bash launch.sh configs/<run>.yaml --slurm --partition <p> --account <a> [--time 04:00:00] [--gpus-per-node 8] [-- <sbatch args>]
#
# Setup is idempotent: every step skips itself when its output exists. Machine
# settings (WORKROOT, ports, WANDB_API_KEY, APPTAINER_*) are environment
# variables; see README.md. Before the first run: stage the task directory and
# `bash prepare_images.sh <task_dir>` (needs the registry; do it on a login node).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PROJECT_ROOT="$(cd -- "${HERE}/../.." && pwd)"

CONFIG=""; DRY_RUN=0; SETUP_ONLY=0; SLURM=0; PARTITION=""; ACCOUNT=""; TIME="04:00:00"; GPUS_PER_NODE_OPT=8; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        --slurm) SLURM=1; shift ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE_OPT="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) CONFIG="$1"; shift ;;
    esac
done
[ -f "${CONFIG}" ] || { echo "usage: launch.sh <run-config.yaml> [--dry-run|--setup-only|--slurm ...]" >&2; exit 2; }
CONFIG="$(cd -- "$(dirname -- "${CONFIG}")" && pwd)/$(basename -- "${CONFIG}")"

# shellcheck source=./internal/setup/common.sh
source "${HERE}/internal/setup/common.sh"
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
mkdir -p "${WORKROOT}"
eval "$(config_python "${HERE}/internal/render.py" env "${CONFIG}")"
echo "${SUMMARY}"

# ── Slurm: submit and exit ─────────────────────────────────────────────────
if [ "${SLURM}" = 1 ]; then
    LOG_DIR="${SLURM_LOG_DIR:-${WORKROOT}/joblogs}"; mkdir -p "${LOG_DIR}"
    args=(--job-name "harbor-grpo-${RUN_ID}" --nodes "${NUM_NODES}" --ntasks-per-node 1 --gpus-per-node "${GPUS_PER_NODE_OPT}"
          --time "${TIME}" --output "${LOG_DIR}/%j-%x.log" --export ALL)
    [ -n "${PARTITION}" ] && args+=(--partition "${PARTITION}")
    [ -n "${ACCOUNT}" ] && args+=(--account "${ACCOUNT}")
    echo "sbatch ${args[*]} ${EXTRA[*]:-} (logs: ${LOG_DIR})"
    exec sbatch "${args[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --wrap "bash ${HERE}/internal/head_entry.sh ${CONFIG}"
fi
if [ -n "${SLURM_JOB_NUM_NODES:-}" ] && [ "${SLURM_JOB_NUM_NODES}" != "${NUM_NODES}" ]; then
    die "config cluster.num_nodes=${NUM_NODES} but the slurm allocation has ${SLURM_JOB_NUM_NODES} nodes"
fi
[ -d "${TASKS_DIR}" ] || die "tasks.dir does not exist: ${TASKS_DIR} (stage it first: harbor datasets download / hf download; see README)"

# ── Placement ──────────────────────────────────────────────────────────────
# Caches default under WORKROOT; ones you export yourself are honored.
export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKROOT}/uv_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORKROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${WORKROOT}/apptainer_tmp}"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${RUN_DIR}"
# Per-run copy of the setup environment (PATH, CUDA, caches): setup/ scripts append
# to it, run.sh and the worker nodes source it. Jobs share a WORKROOT concurrently,
# so nothing job-written lives at WORKROOT level.
export ENV_FILE="${RUN_DIR}/env.sh"
export DRY_RUN SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"

# ── Environment (idempotent; serialized across concurrent jobs on one WORKROOT) ──
if [ "${DRY_RUN}" = 0 ]; then
    setup_lock_acquire "${WORKROOT}/.setup.lock"
    : > "${ENV_FILE}"
    for step in preflight install_python_stack ensure_cuda_userspace ensure_apptainer ensure_checkouts ensure_training_stack; do
        # shellcheck disable=SC1090
        source "${HERE}/internal/setup/${step}.sh"
    done
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
else
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
    [ -x "${PYTHON_BIN}" ] && "${PYTHON_BIN}" -c 'import yaml' 2>/dev/null || PYTHON_BIN="$(command -v python3)"
    export PYTHON_BIN
fi

# ── Tasks -> prompts (images must already be pulled: prepare_images.sh) ────
log "tasks"
select=(--mount-root "${TASKS_MOUNT_ROOT}" --seed "${TASKS_SEED}")
[ -n "${TASKS_N}" ] && select+=(--n "${TASKS_N}")
[ -n "${TASK_IDS_FILE}" ] && select+=(--task-ids-file "${TASK_IDS_FILE}")
[ -n "${EXCLUDE_IDS_FILE}" ] && select+=(--exclude-ids-file "${EXCLUDE_IDS_FILE}")
[ "${DRY_RUN}" = 1 ] || select+=(--image-dir "${APPTAINER_IMAGE_DIR}")
"${PYTHON_BIN}" "${HERE}/internal/prepare_tasks.py" --tasks-dir "${TASKS_DIR}" --output-jsonl "${RUN_DIR}/train.jsonl" "${select[@]}"

if [ "${DRY_RUN}" = 0 ]; then
    log "harness ${HARNESS}${HARNESS_CLI_VERSION:+ @ ${HARNESS_CLI_VERSION}}"
    bash "${HERE}/internal/prepare_harness.sh" "${HARNESS_DIR}" "${HARNESS}" "${HARNESS_CLI_VERSION}"
    case "${HF_CHECKPOINT}" in
        /*|./*|../*|~*) ;;
        *)  log "HF snapshot ${HF_CHECKPOINT}"
            # The converter reads *.safetensors from the local cache and does not download.
            "$(dirname "${PYTHON_BIN}")/hf" download "${HF_CHECKPOINT}" >/dev/null && info "present in ${HF_HOME}" ;;
    esac
    if [ ! -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
        log "HF -> torch_dist conversion"
        bash "${HERE}/internal/convert_weights.sh"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        "${PYTHON_BIN}" -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)' 2>/dev/null || true
    fi
    setup_lock_release
fi

# ── Polar configs + slime arguments ────────────────────────────────────────
log "render"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
[ "${GPUS_PER_NODE}" -ge 1 ] 2>/dev/null || export GPUS_PER_NODE=8
"${PYTHON_BIN}" "${HERE}/internal/render.py" run "${CONFIG}"
if [ "${DRY_RUN}" = 1 ]; then
    echo "--- ${RUN_DIR}/polar_config.yaml ---"; cat "${RUN_DIR}/polar_config.yaml"
    echo "--- ${RUN_DIR}/train_args.sh ---"; cat "${RUN_DIR}/train_args.sh"
    exit 0
fi
[ "${SETUP_ONLY}" = 0 ] || { echo "setup done (--setup-only): ${RUN_DIR}"; exit 0; }
log "run.sh"
exec bash "${HERE}/internal/run.sh"
