#!/usr/bin/env bash
# GRPO on a directory of Harbor tasks with Polar + Miles (LoRA or full fine-tune), from one run config.
#
#   bash launch.sh configs/<run>.yaml                  # run here (inside a slurm allocation; head node)
#   bash launch.sh configs/<run>.yaml --dry-run        # resolve config, build prompts, render; no GPUs
#   bash launch.sh configs/<run>.yaml --setup-only     # Miles image, polar venv, harness, prompts; no training
#
# Same flow as harbor_slime_grpo/launch.sh, with the trainer in the pinned Miles
# image (examples/miles_runtime) instead of a locked venv: Polar (rollout server,
# gateways, sandboxes) runs on the host with a small venv, Ray + Miles + SGLang run
# inside the image on every node. Shared pieces (task prep, harness, Polar templates)
# come from ../harbor_slime_grpo. Multi-node runs start from internal/head_entry.sh.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PROJECT_ROOT="$(cd -- "${HERE}/../.." && pwd)"
SHARED="${PROJECT_ROOT}/examples/harbor_slime_grpo"

CONFIG=""; DRY_RUN=0; SETUP_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown option $1" >&2; exit 2 ;;
        *) CONFIG="$1"; shift ;;
    esac
done
[ -f "${CONFIG}" ] || { echo "usage: launch.sh <run-config.yaml> [--dry-run|--setup-only]" >&2; exit 2; }
CONFIG="$(cd -- "$(dirname -- "${CONFIG}")" && pwd)/$(basename -- "${CONFIG}")"

# shellcheck source=../harbor_slime_grpo/internal/setup/common.sh
source "${SHARED}/internal/setup/common.sh"
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
mkdir -p "${WORKROOT}"
PYTHON_BIN="${POLAR_PYTHON:-${WORKROOT}/polar_venv/bin/python}"; export PYTHON_BIN  # config_python prefers it once it exists
eval "$(config_python "${HERE}/internal/render.py" env "${CONFIG}")"
echo "${SUMMARY}"
if [ -n "${SLURM_JOB_NUM_NODES:-}" ] && [ "${SLURM_JOB_NUM_NODES}" != "${NUM_NODES}" ]; then
    die "config cluster.num_nodes=${NUM_NODES} but the slurm allocation has ${SLURM_JOB_NUM_NODES} nodes"
fi
[ -d "${TASKS_DIR}" ] || die "tasks.dir does not exist: ${TASKS_DIR}"

export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORKROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${WORKROOT}/apptainer_tmp}"
mkdir -p "${HF_HOME}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}" "${RUN_DIR}"
export ENV_FILE="${RUN_DIR}/env.sh" DRY_RUN

if [ "${DRY_RUN}" = 0 ]; then
    setup_lock_acquire "${WORKROOT}/.setup.lock"
    : > "${ENV_FILE}"
    emit_export PROJECT_ROOT "${PROJECT_ROOT}"
    emit_export HF_HOME "${HF_HOME}"
    # shellcheck disable=SC1091
    source "${HERE}/internal/setup/ensure_miles_runtime.sh"
    # Task sandboxes use the same Apptainer the runtime installed (or found on PATH).
    NEED_APPTAINER=0
    export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-${MILES_APPTAINER_BIN}}"
    # shellcheck disable=SC1091
    source "${SHARED}/internal/setup/ensure_apptainer.sh"
    # shellcheck disable=SC1091
    source "${HERE}/internal/setup/ensure_polar_venv.sh"
    PYTHON_BIN="${POLAR_PYTHON}"
fi

log "tasks"
select=(--mount-root "${TASKS_MOUNT_ROOT}" --seed "${TASKS_SEED}")
[ -n "${TASKS_N}" ] && select+=(--n "${TASKS_N}")
[ -n "${TASK_IDS_FILE}" ] && select+=(--task-ids-file "${TASK_IDS_FILE}")
[ -n "${EXCLUDE_IDS_FILE}" ] && select+=(--exclude-ids-file "${EXCLUDE_IDS_FILE}")
if [ "${DRY_RUN}" = 0 ]; then
    # Task images live in this work root's image dir. Images another work root already
    # has (APPTAINER_IMAGE_DIR from the cluster profile, e.g. a shared SIF cache) are
    # symlinked, never written; only what is missing everywhere is pulled, into our dir.
    # One task-image dir per cluster for every Miles run (all agents): <user root>/miles/shared/harbor_sif_images.
    # Images other dirs already hold (the cluster profile's SIF cache, this work root's older dir) are symlinked in,
    # never moved; anything still missing is pulled into it. MILES_IMAGE_DIR overrides.
    SHARED_IMAGE_DIRS=("${APPTAINER_IMAGE_DIR:-}" "${WORKROOT}/harbor_sif_images")
    default_dir="${WORKROOT}/harbor_sif_images"
    [ -n "${CLUSTER_USER_ROOT:-}" ] && default_dir="${CLUSTER_USER_ROOT}/miles/shared/harbor_sif_images"
    export APPTAINER_IMAGE_DIR="${MILES_IMAGE_DIR:-${default_dir}}"
    SHARED_IMAGE_DIR="${SHARED_IMAGE_DIRS[0]}"
    mkdir -p "${APPTAINER_IMAGE_DIR}"
    log "task images -> ${APPTAINER_IMAGE_DIR}${SHARED_IMAGE_DIR:+ (links into ${SHARED_IMAGE_DIR})}"
    to_pull="$(mktemp)"
    config_python "${HERE}/internal/stage_images.py" --tasks-dir "${TASKS_DIR}" --image-dir "${APPTAINER_IMAGE_DIR}" \
        --shared-dir "${SHARED_IMAGE_DIRS[0]}" --shared-dir "${SHARED_IMAGE_DIRS[1]}" "${select[@]:2}" > "${to_pull}"
    # Registry credentials in the environment (the cluster layer maps a GitLab token to APPTAINER_DOCKER_*)
    # belong to that registry only: Docker Hub refs (no registry host) are pulled anonymously.
    pull_one() {
        local ref="$1" sif="$2" creds=() try
        case "${ref%%/*}" in *.*|*:*|localhost) ;; *) creds=(env -u APPTAINER_DOCKER_USERNAME -u APPTAINER_DOCKER_PASSWORD -u SINGULARITY_DOCKER_USERNAME -u SINGULARITY_DOCKER_PASSWORD) ;; esac
        for try in 1 2 3; do   # registry streams drop now and then (INTERNAL_ERROR mid-layer)
            echo "  pulling ${ref} -> ${sif} (try ${try})"
            "${creds[@]}" "${POLAR_APPTAINER_BIN}" pull --force "${APPTAINER_IMAGE_DIR}/${sif}.part.$$" "docker://${ref}" >/dev/null 2>"${APPTAINER_IMAGE_DIR}/${sif}.pull.log" \
                && mv "${APPTAINER_IMAGE_DIR}/${sif}.part.$$" "${APPTAINER_IMAGE_DIR}/${sif}" && return 0
            sleep $((30 * try))
        done
        echo "  FAILED ${ref} (${APPTAINER_IMAGE_DIR}/${sif}.pull.log)"
    }
    export -f pull_one; export APPTAINER_IMAGE_DIR POLAR_APPTAINER_BIN
    NCPU="$(nproc 2>/dev/null || echo 8)"; JOBS="${APPTAINER_JOBS:-6}"
    export APPTAINER_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS:-$(( NCPU / JOBS > 0 ? NCPU / JOBS : 1 ))}"
    cut -f1,2 "${to_pull}" | tr '\t' ' ' | xargs -r -P "${JOBS}" -L 1 bash -c 'pull_one "$0" "$1"'
    # Tasks whose image still could not be staged are left out of this run (listed), not fatal.
    config_python "${HERE}/internal/stage_images.py" --tasks-dir "${TASKS_DIR}" --image-dir "${APPTAINER_IMAGE_DIR}" \
        --shared-dir "${SHARED_IMAGE_DIRS[0]}" --shared-dir "${SHARED_IMAGE_DIRS[1]}" "${select[@]:2}" > "${to_pull}"
    if [ -s "${to_pull}" ]; then
        skip="${RUN_DIR}/tasks_without_image.txt"
        cut -f3 "${to_pull}" | tr ',' '\n' > "${skip}"
        [ -n "${EXCLUDE_IDS_FILE}" ] && cat "${EXCLUDE_IDS_FILE}" >> "${skip}"
        echo "  WARNING: $(cut -f3 "${to_pull}" | tr ',' '\n' | wc -l) task(s) without an image are excluded: ${skip}"
        sel=(); skip_next=0
        for a in "${select[@]}"; do   # replace any --exclude-ids-file with the merged list
            if [ "${skip_next}" = 1 ]; then skip_next=0; continue; fi
            if [ "${a}" = --exclude-ids-file ]; then skip_next=1; continue; fi
            sel+=("${a}")
        done
        select=("${sel[@]}" --exclude-ids-file "${skip}")
    fi
    select+=(--image-dir "${APPTAINER_IMAGE_DIR}")
fi
config_python "${SHARED}/internal/prepare_tasks.py" --tasks-dir "${TASKS_DIR}" --output-jsonl "${RUN_DIR}/train.jsonl" "${select[@]}"
if [ -n "${EVAL_TASK_IDS_FILE}" ]; then   # held-out eval set from the same task dir
    eval_select=(--mount-root "${TASKS_MOUNT_ROOT}" --task-ids-file "${EVAL_TASK_IDS_FILE}")
    if [ "${DRY_RUN}" = 0 ]; then
        to_pull_eval="$(mktemp)"
        config_python "${HERE}/internal/stage_images.py" --tasks-dir "${TASKS_DIR}" --image-dir "${APPTAINER_IMAGE_DIR}" \
            --shared-dir "${SHARED_IMAGE_DIRS[0]}" --shared-dir "${SHARED_IMAGE_DIRS[1]}" --task-ids-file "${EVAL_TASK_IDS_FILE}" > "${to_pull_eval}"
        cut -f1,2 "${to_pull_eval}" | tr '\t' ' ' | xargs -r -P "${JOBS:-6}" -L 1 bash -c 'pull_one "$0" "$1"'
        eval_select+=(--image-dir "${APPTAINER_IMAGE_DIR}")
    fi
    config_python "${SHARED}/internal/prepare_tasks.py" --tasks-dir "${TASKS_DIR}" --output-jsonl "${RUN_DIR}/eval.jsonl" "${eval_select[@]}"
fi

if [ "${DRY_RUN}" = 0 ]; then
    log "harness ${HARNESS}${HARNESS_CLI_VERSION:+ @ ${HARNESS_CLI_VERSION}} in ${HARNESS_DIR}"
    case "${HARNESS_DIR}" in
        "${WORKROOT}"/*) bash "${SHARED}/internal/prepare_harness.sh" "${HARNESS_DIR}" "${HARNESS}" "${HARNESS_CLI_VERSION}" ;;
        *)  # Another work root's harness dir: use it read-only, never reinstall into it.
            bin="${HARNESS}"; [ "${bin}" = mini_swe_agent ] && bin=mini-swe-agent
            [ -x "${HARNESS_DIR}/bin/${bin}" ] || die "harness.dir ${HARNESS_DIR} has no bin/${bin} (point harness.dir inside WORKROOT to install one)"
            info "read-only: $(PATH="${HARNESS_DIR}/node/bin:${PATH}" "${HARNESS_DIR}/bin/${bin}" --version 2>&1 | tail -1)" ;;
    esac
    case "${HF_CHECKPOINT}" in
        /*) [ -f "${HF_CHECKPOINT}/config.json" ] || die "model.hf_checkpoint ${HF_CHECKPOINT} has no config.json" ;;
        *)  log "HF snapshot ${HF_CHECKPOINT}"
            "$(dirname "${POLAR_PYTHON}")/hf" download "${HF_CHECKPOINT}" >/dev/null && info "present in ${HF_HOME}" ;;
    esac
    if [ "${LORA}" = 0 ] && [ ! -f "${TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]; then
        log "HF -> torch_dist conversion (full fine-tune base) -> ${TORCH_DIST_DIR}"
        bash "${HERE}/internal/convert_weights.sh"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        "${POLAR_PYTHON}" -c 'import wandb' 2>/dev/null && "${POLAR_PYTHON}" -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)' 2>/dev/null || true
    fi
    setup_lock_release
fi

[ "${SETUP_ONLY}" = 0 ] || { echo "setup done (--setup-only): ${RUN_DIR}"; exit 0; }
log "render"
export GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
[ "${GPUS_PER_NODE}" -ge 1 ] 2>/dev/null || export GPUS_PER_NODE=8
config_python "${HERE}/internal/render.py" run "${CONFIG}"
if [ "${DRY_RUN}" = 1 ]; then
    echo "--- ${RUN_DIR}/polar_config.yaml ---"; cat "${RUN_DIR}/polar_config.yaml"
    echo "--- ${RUN_DIR}/topology.yaml ---"; cat "${RUN_DIR}/topology.yaml"
    echo "--- ${RUN_DIR}/train_args.sh ---"; cat "${RUN_DIR}/train_args.sh"
    exit 0
fi
log "run.sh"
exec bash "${HERE}/internal/run.sh"
