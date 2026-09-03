#!/usr/bin/env bash
# Config-driven GRPO training on Harbor tasks with Polar + Slime.
#
#   bash examples/harbor_slime_grpo/launch.sh configs/<run>.yaml [--dry-run]
#
# One run config describes everything: the task directory and subset, the agent
# harness, the model, parallelism and algorithm knobs. This script
#   1. sets up the environment (idempotent; SETUP_ENV=0 to bring your own),
#   2. turns the task directory into a prompt JSONL + per-task SIF images and
#      builds the harness directory,
#   3. downloads and converts the checkpoint,
#   4. renders the Polar templates and hands off to run.sh (Polar services,
#      Ray, Slime), on the head node when multi-node (multinode/head_entry.sh).
# --dry-run stops after rendering: it prints the environment run.sh would see and
# the rendered Polar config, without touching GPUs, images or checkpoints.
#
# Everything created lands under WORKROOT (default <repo>/tmp). Cache variables
# you export (HF_HOME, UV_CACHE_DIR, APPTAINER_CACHEDIR, ...) are respected.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_CONFIG=""; DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) RUN_CONFIG="${arg}" ;;
    esac
done
[ -n "${RUN_CONFIG}" ] || { echo "usage: launch.sh <run-config.yaml> [--dry-run]" >&2; exit 2; }
[ -f "${RUN_CONFIG}" ] || { echo "ERROR: run config not found: ${RUN_CONFIG}" >&2; exit 2; }
RUN_CONFIG="$(cd -- "$(dirname -- "${RUN_CONFIG}")" && pwd)/$(basename -- "${RUN_CONFIG}")"
CONFIG_DIR="$(dirname -- "${RUN_CONFIG}")"

cd "${PROJECT_ROOT}"
export PROJECT_ROOT
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
export ENV_FILE="${ENV_FILE:-${WORKROOT}/env.sh}"
export SETUP_ENV="${SETUP_ENV:-1}"
export TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
mkdir -p "${WORKROOT}"
export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKROOT}/uv_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORKROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${WORKROOT}/apptainer_tmp}"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# shellcheck source=./setup/common.sh
source "${SCRIPT_DIR}/setup/common.sh"

# ── 0. Python for config parsing (the venv once it exists, else system) ────
CFG_PYTHON="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
[ -x "${CFG_PYTHON}" ] || CFG_PYTHON="$(command -v python3)"
"${CFG_PYTHON}" -c 'import yaml' 2>/dev/null || die "${CFG_PYTHON} lacks pyyaml; run with the Polar venv or install pyyaml"

# ── 1. Run config → environment ────────────────────────────────────────────
# Every key has a default; only tasks.dir and name are required. Paths in the
# config that are relative resolve against the config file's directory.
eval "$("${CFG_PYTHON}" - "${RUN_CONFIG}" "${CONFIG_DIR}" <<'PY'
import os, shlex, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
cfg_dir = sys.argv[2]
def path(v):
    if not v: return ""
    v = os.path.expanduser(os.path.expandvars(str(v)))   # ${WORKROOT} etc. from the environment
    return v if os.path.isabs(v) else os.path.normpath(os.path.join(cfg_dir, v))
t, h, m, tr = cfg.get("tasks", {}), cfg.get("harness", {}), cfg.get("model", {}), cfg.get("training", {})
for key in ("name",):
    if key not in cfg: sys.exit(f"run config lacks '{key}'")
if "dir" not in t: sys.exit("run config lacks tasks.dir")
out = {
    "RUN_NAME": cfg["name"],
    "TASKS_DIR": path(t["dir"]),
    "TASKS_MOUNT_ROOT": path(t.get("mount_root", "")),
    "TASKS_N": t.get("n", ""),
    "TASKS_SEED": t.get("seed", 0),
    "TASK_IDS_FILE": path(t.get("task_ids_file", "")),
    "EXCLUDE_IDS_FILE": path(t.get("exclude_ids_file", "")),
    "HARNESS": h.get("name", "mini_swe_agent"),
    "HARNESS_MODEL_NAME": h.get("model_name", "openai/gpt-5.4"),
    "HARNESS_DIR_CFG": path(h.get("dir", "")),
    "HF_CHECKPOINT": m.get("hf_checkpoint", "Qwen/Qwen3.5-9B"),
    "MODEL_ARGS_FILE": m.get("model_args_file", "model_args_9b.sh"),
    "TORCH_DIST_DIR_CFG": path(m.get("torch_dist_dir", "")),
    "EOT_TOKEN_ID": m.get("end_of_turn_token_id", 248046),
    "TRAIN_SCRIPT": "train.py" if tr.get("sync", True) else "train_async.py",
    "TP_SIZE": tr.get("tp_size", 4),
    "CONTEXT_PARALLEL_SIZE": tr.get("context_parallel_size", 2),
    "ACTOR_NUM_GPUS": tr.get("actor_num_gpus", 8),
    "ROLLOUT_BATCH_SIZE": tr.get("rollout_batch_size", 8),
    "N_SAMPLES_PER_PROMPT": tr.get("n_samples_per_prompt", 16),
    "NUM_EPOCH": tr.get("num_epoch", 50),
    "MAX_TOKENS_PER_GPU": tr.get("max_tokens_per_gpu", 16384),
    "SGLANG_CONTEXT_LENGTH": tr.get("sglang_context_length", 32768),
    "ROLLOUT_MAX_PROMPT_LEN": tr.get("rollout_max_prompt_len", 24000),
    "ROLLOUT_MAX_RESPONSE_LEN": tr.get("rollout_max_response_len", 8000),
    "SAVE_INTERVAL": tr.get("save_interval", 5),
    "SESSION_TIMEOUT": tr.get("session_timeout", 3000),
    "REQUEST_TIMEOUT": tr.get("request_timeout", 3600),
    "MAX_RUN_WORKERS": tr.get("max_run_workers", 16),
    "MAX_ASYNC_LEVEL": tr.get("max_async_level", 1),
    "USE_KL_LOSS": int(bool(tr.get("use_kl_loss", False))),
    "GRPO_STD_NORMALIZATION": int(bool(tr.get("grpo_std_normalization", False))),
    "GROUP_ID_SCOPE": tr.get("group_id_scope", "trajectory"),
    "TIMEOUT_REWARD_ZERO": str(bool(tr.get("timeout_reward_zero", True))).lower(),
    "DROP_ZERO_VARIANCE_GROUPS": str(bool(tr.get("drop_zero_variance_groups", True))).lower(),
    "EXTRA_TRAIN_ARGS": tr.get("extra_train_args", ""),
    "WANDB_PROJECT": cfg.get("wandb_project", "harbor-slime-grpo"),
}
for k, v in out.items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
)"
[ -d "${TASKS_DIR}" ] || die "tasks.dir does not exist: ${TASKS_DIR}"
export RUN_ID="${RUN_ID:-${RUN_NAME}}"
ASSET_DIR="${WORKROOT}/harbor_slime_grpo/${RUN_ID}/assets"
mkdir -p "${ASSET_DIR}"
export HARNESS_DIR="${HARNESS_DIR_CFG:-${WORKROOT}/harbor_harness}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${WORKROOT}/harbor_sif_images}"
export HARBOR_DATASET_DIR="${TASKS_MOUNT_ROOT:-${TASKS_DIR}}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR_CFG:-${TORCH_DIST_DIR:-${WORKROOT}/checkpoints/${HF_CHECKPOINT##*/}_torch_dist}}"
export REF_LOAD="${TORCH_DIST_DIR}"
export SAVE_ROOT="${SAVE_ROOT:-${WORKROOT}/ckpt/harbor_slime_grpo}"
export RUN_DIR="${WORKROOT}/harbor_slime_grpo/${RUN_ID}"

# ── 2. Environment (same recipe as the swegym example, kept self-contained) ─
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.3.0}"
MEGATRON_DIR="${MEGATRON_DIR:-${WORKROOT}/Megatron-LM-slime-${SLIME_REF}}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-1dcf0dafa884ad52ffb243625717a3471643e087}"
MEGATRON_PATCH="${MEGATRON_PATCH:-${SLIME_DIR}/docker/patch/latest/megatron.patch}"
MBRIDGE_VERSION="${MBRIDGE_VERSION:-0.15.1}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-1}"
APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-1}"
PREPARE_IMAGES="${PREPARE_IMAGES:-1}"
PREPARE_HARNESS="${PREPARE_HARNESS:-1}"
APPTAINER_PREPARE_JOBS="${APPTAINER_PREPARE_JOBS:-4}"
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"
RUN_TRAINING="${RUN_TRAINING:-1}"

if [ "${DRY_RUN}" = 0 ]; then
    if [ "${SETUP_ENV}" = 1 ]; then
        : > "${ENV_FILE}"
        # shellcheck source=./setup/preflight.sh
        source "${SCRIPT_DIR}/setup/preflight.sh"
        # shellcheck source=./setup/install_python_stack.sh
        source "${SCRIPT_DIR}/setup/install_python_stack.sh"
        # shellcheck source=./setup/ensure_cuda_userspace.sh
        source "${SCRIPT_DIR}/setup/ensure_cuda_userspace.sh"
        # shellcheck source=./setup/ensure_apptainer.sh
        source "${SCRIPT_DIR}/setup/ensure_apptainer.sh"
    else
        # shellcheck disable=SC1090
        [ -f "${ENV_FILE}" ] && source "${ENV_FILE}"
        PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
        [ -x "${PYTHON_BIN}" ] || die "SETUP_ENV=0 but PYTHON_BIN=${PYTHON_BIN} is not executable"
        export PYTHON_BIN
        export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || command -v singularity || true)}"
        [ -x "${POLAR_APPTAINER_BIN}" ] || die "SETUP_ENV=0 but no apptainer binary (set POLAR_APPTAINER_BIN)"
    fi
    PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
    export PATH="${PYTHON_BIN_DIR}:${PATH}"
    UV_PIP=(uv pip install --python "${PYTHON_BIN}" --torch-backend="${TORCH_BACKEND}" --prerelease=allow)

    clone_retry() {   # clone_retry NAME REPO REF DEST
        local name="$1" repo="$2" ref="$3" dest="$4" i
        if [ -d "${dest}/.git" ]; then info "${name} checkout exists: ${dest}"; return 0; fi
        [ -e "${dest}" ] && die "${name} path exists but is not a git checkout: ${dest}"
        for i in 1 2 3 4 5; do
            if [[ "${ref}" =~ ^[0-9a-f]{40}$ ]]; then
                mkdir -p "${dest}" && git -C "${dest}" init -q && git -C "${dest}" remote add origin "${repo}" \
                  && git -C "${dest}" fetch -q --depth 1 origin "${ref}" && git -C "${dest}" checkout -q FETCH_HEAD && return 0
            else
                git clone -q --branch "${ref}" --depth 1 "${repo}" "${dest}" && return 0
            fi
            info "clone of ${name} failed (try ${i}/5); retrying in 20s"; rm -rf "${dest}"; sleep 20
        done
        die "could not clone ${name} from ${repo}"
    }
    log "checkouts"
    clone_retry Slime "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
    clone_retry Megatron-LM "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"
    if [ -f "${MEGATRON_PATCH}" ]; then
        if git -C "${MEGATRON_DIR}" apply --reverse --check "${MEGATRON_PATCH}" >/dev/null 2>&1; then
            info "slime megatron.patch already applied"
        else
            git -C "${MEGATRON_DIR}" apply --3way "${MEGATRON_PATCH}" && info "applied slime megatron.patch"
        fi
    fi
    SLIME_DIR="${SLIME_DIR}" bash "${PROJECT_ROOT}/scripts/patch/patch_slime_router_tokens.sh"
    export SLIME_DIR MEGATRON_DIR

    if [ "${INSTALL_EDITABLE}" = 1 ]; then
        log "editable installs"
        "${UV_PIP[@]}" -e "."
        "${UV_PIP[@]}" -e "${SLIME_DIR}"
        "${UV_PIP[@]}" -e "${MEGATRON_DIR}"
        "${UV_PIP[@]}" --no-deps "mbridge==${MBRIDGE_VERSION}"
        "${UV_PIP[@]}" pyarrow huggingface_hub   # datasets/*.py
    fi
    if [ "${INSTALL_TRAINING_STACK}" = 1 ] && [ "${SETUP_ENV}" = 1 ]; then
        # shellcheck source=./setup/ensure_training_stack.sh
        source "${SCRIPT_DIR}/setup/ensure_training_stack.sh"
    fi
    if [ "${APPLY_SGLANG_PATCH}" = 1 ]; then
        log "sglang token-metadata patch"
        VIRTUAL_ENV="$(dirname "${PYTHON_BIN_DIR}")" bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_0513_token_metadata.sh"
    fi
else
    PYTHON_BIN="${CFG_PYTHON}"
fi

# ── 3. Tasks → prompts, images, harness ────────────────────────────────────
log "tasks"
SELECT=(--mount-root "${HARBOR_DATASET_DIR}" --seed "${TASKS_SEED}")
[ -n "${TASKS_N}" ] && SELECT+=(--n "${TASKS_N}")
[ -n "${TASK_IDS_FILE}" ] && SELECT+=(--task-ids-file "${TASK_IDS_FILE}")
[ -n "${EXCLUDE_IDS_FILE}" ] && SELECT+=(--exclude-ids-file "${EXCLUDE_IDS_FILE}")
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_tasks.py" --tasks-dir "${TASKS_DIR}" \
    --output-jsonl "${ASSET_DIR}/train.jsonl" --output-images "${ASSET_DIR}/images.txt" "${SELECT[@]}"
export PROMPT_DATA="${ASSET_DIR}/train.jsonl"

if [ "${DRY_RUN}" = 0 ]; then
    if [ "${PREPARE_IMAGES}" = 1 ]; then
        log "task images (SIF)"
        bash "${SCRIPT_DIR}/prepare_images.sh" "${ASSET_DIR}/images.txt" "${APPTAINER_IMAGE_DIR}" "${APPTAINER_PREPARE_JOBS}"
    fi
    if [ "${PREPARE_HARNESS}" = 1 ]; then
        log "harness ${HARNESS}"
        bash "${SCRIPT_DIR}/prepare_harness.sh" "${HARNESS_DIR}" "${HARNESS}"
    fi

    # ── 4. Checkpoint ──────────────────────────────────────────────────────
    case "${HF_CHECKPOINT}" in
        /*|./*|../*|~*) ;;
        *)  log "HF snapshot ${HF_CHECKPOINT}"
            # mbridge reads *.safetensors from the local cache and does not download.
            "${PYTHON_BIN_DIR}/hf" download "${HF_CHECKPOINT}" >/dev/null && info "present in ${HF_HOME}" ;;
    esac
    if [ "${CONVERT_WEIGHTS}" = 1 ] || { [ "${CONVERT_WEIGHTS}" = auto ] && [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; }; then
        log "HF → torch_dist conversion"
        MODEL_ARGS_FILE="${MODEL_ARGS_FILE}" HF_CHECKPOINT="${HF_CHECKPOINT}" TORCH_DIST_DIR="${TORCH_DIST_DIR}" \
            bash "${SCRIPT_DIR}/convert_weights.sh"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        "${PYTHON_BIN}" -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)' 2>/dev/null || true
    fi
fi

# ── 5. Render the Polar templates (@TOKENS@ + agent.settings) ─────────────
log "polar templates"
export POLAR_CONFIG_TEMPLATE="${ASSET_DIR}/polar_config.yaml"
export TOPOLOGY_TEMPLATE="${ASSET_DIR}/topology.yaml"
"${PYTHON_BIN}" - "${RUN_CONFIG}" "${SCRIPT_DIR}" "${ASSET_DIR}" <<'PY'
import os, sys, yaml
cfg_path, tpl_dir, out_dir = sys.argv[1:4]
cfg = yaml.safe_load(open(cfg_path)) or {}
env = os.environ
tokens = {
    "@HARNESS@": env["HARNESS"], "@HARNESS_MODEL_NAME@": env["HARNESS_MODEL_NAME"],
    "@HARNESS_DIR@": env["HARNESS_DIR"], "@HARBOR_DATASET_DIR@": env["HARBOR_DATASET_DIR"],
    "@RUN_NAME@": env["RUN_NAME"], "@GROUP_ID_SCOPE@": env["GROUP_ID_SCOPE"],
}
numeric = {
    "@SESSION_TIMEOUT@": int(env["SESSION_TIMEOUT"]), "@REQUEST_TIMEOUT@": int(env["REQUEST_TIMEOUT"]),
    "@MAX_ASYNC_LEVEL@": int(env["MAX_ASYNC_LEVEL"]), "@EOT_TOKEN_ID@": int(env["EOT_TOKEN_ID"]),
    "@MAX_RUN_WORKERS@": int(env["MAX_RUN_WORKERS"]),
    "@TIMEOUT_REWARD_ZERO@": env["TIMEOUT_REWARD_ZERO"] == "true",
    "@DROP_ZERO_VARIANCE_GROUPS@": env["DROP_ZERO_VARIANCE_GROUPS"] == "true",
}
def fill(node):
    if isinstance(node, dict): return {k: fill(v) for k, v in node.items()}
    if isinstance(node, list): return [fill(v) for v in node]
    if isinstance(node, str):
        if node in numeric: return numeric[node]
        for k, v in tokens.items(): node = node.replace(k, v)
    return node
class Dumper(yaml.SafeDumper):  # keep ${VARS} unquoted-safe and lists indented
    pass
for name in ("polar_config.yaml", "topology.yaml"):
    doc = fill(yaml.safe_load(open(os.path.join(tpl_dir, name))))
    if name == "polar_config.yaml":
        doc["polar_task_template"]["agent"]["settings"] = (cfg.get("harness") or {}).get("settings") or {}
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(f"# rendered by launch.sh from {os.path.join(tpl_dir, name)} and {cfg_path}\n")
        yaml.dump(doc, f, Dumper=Dumper, sort_keys=False, default_flow_style=False, width=200)
print(f"rendered polar_config.yaml + topology.yaml -> {out_dir}")
PY
chmod 600 "${POLAR_CONFIG_TEMPLATE}"

# ── 6. Hand off ────────────────────────────────────────────────────────────
export HARNESS HARNESS_DIR TRAIN_SCRIPT TP_SIZE CONTEXT_PARALLEL_SIZE ACTOR_NUM_GPUS
export ROLLOUT_BATCH_SIZE N_SAMPLES_PER_PROMPT NUM_EPOCH MAX_TOKENS_PER_GPU SGLANG_CONTEXT_LENGTH
export ROLLOUT_MAX_PROMPT_LEN ROLLOUT_MAX_RESPONSE_LEN SAVE_INTERVAL USE_KL_LOSS GRPO_STD_NORMALIZATION EXTRA_TRAIN_ARGS
export HF_CHECKPOINT MODEL_ARGS_FILE WANDB_PROJECT WANDB_GROUP="${WANDB_GROUP:-${RUN_NAME}}"
export PREPARE_IMAGES=0

cat <<INFO

Run:        ${RUN_NAME} (RUN_ID ${RUN_ID})
Tasks:      ${TASKS_DIR} -> $(wc -l < "${PROMPT_DATA}") prompts${TASKS_N:+ (n=${TASKS_N}, seed ${TASKS_SEED})}; $(wc -l < "${ASSET_DIR}/images.txt") image(s)
Harness:    ${HARNESS} from ${HARNESS_DIR}
Model:      ${HF_CHECKPOINT} (${MODEL_ARGS_FILE}) TP${TP_SIZE}xCP${CONTEXT_PARALLEL_SIZE} on ${ACTOR_NUM_GPUS} GPUs; trace cap $((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE)) = sglang ctx ${SGLANG_CONTEXT_LENGTH}
Batch:      ${ROLLOUT_BATCH_SIZE} prompts x ${N_SAMPLES_PER_PROMPT} samples, ${NUM_EPOCH} epochs; ${TRAIN_SCRIPT}; KL=${USE_KL_LOSS} std_norm=${GRPO_STD_NORMALIZATION} scope=${GROUP_ID_SCOPE}
Templates:  ${POLAR_CONFIG_TEMPLATE}, ${TOPOLOGY_TEMPLATE}
Save:       ${SAVE_ROOT}/${RUN_ID}
INFO

if [ "${DRY_RUN}" = 1 ]; then
    echo "--- environment for run.sh ---"
    # Never echo credentials (WANDB_API_KEY etc. may be in the caller's shell).
    env | grep -E '^(RUN_|TASKS_|HARNESS|TRAIN_SCRIPT|TP_SIZE|CONTEXT_PARALLEL|ACTOR_|ROLLOUT_|N_SAMPLES|NUM_EPOCH|MAX_TOKENS|SGLANG_|SAVE_|USE_KL|GRPO_|EXTRA_TRAIN|HF_CHECKPOINT|MODEL_ARGS|TORCH_DIST|REF_LOAD|PROMPT_DATA|POLAR_CONFIG|TOPOLOGY|APPTAINER_IMAGE|HARBOR_DATASET|WANDB_PROJECT|WANDB_GROUP)=' \
        | grep -viE 'key|token|secret|password' | sort
    echo "--- rendered polar_config.yaml ---"
    cat "${POLAR_CONFIG_TEMPLATE}"
    exit 0
fi
[ "${RUN_TRAINING}" = 1 ] || exit 0
log "run.sh"
exec bash "${SCRIPT_DIR}/run.sh"
