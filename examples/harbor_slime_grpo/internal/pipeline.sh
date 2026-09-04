#!/usr/bin/env bash
# Pipeline for the Harbor Slime GRPO example. Not run by hand: launch.sh (or
# head_entry.sh on slurm) exports the run config as environment variables and
# execs this.
#
# Order: preflight → python stack (uv sync of setup/stack/uv.lock; slime and
# sglang come from the pinned polar forks) → CUDA user space → apptainer →
# slime checkout (train scripts, conversion tool, megatron.patch) + Megatron
# checkout (patched, editable) → training stack → tasks → images + harness →
# HF snapshot → weight conversion → Polar templates → run.sh.
# Every step is idempotent; re-running resumes where the previous run stopped.
# DRY_RUN=1 (launch.sh --dry-run) skips everything that touches GPUs, images or
# checkpoints and still builds the prompt list and renders the templates.
#
# Placement: WORKROOT (default: <repo>/tmp) holds everything this script
# creates. Cache variables you already export (HF_HOME, UV_CACHE_DIR,
# APPTAINER_CACHEDIR, ...) are left alone; only unset ones are pointed under
# WORKROOT. SETUP_ENV=0 skips preflight and all setup/ scripts (bring your own
# venv with CUDA torch, TE, and apptainer; PYTHON_BIN must then point at it).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
export PROJECT_ROOT
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
# ENV_FILE (the persisted setup environment) is per run; set below once RUN_DIR is known.
export SETUP_ENV="${SETUP_ENV:-1}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${WORKROOT}"

# ── Caches: fill only what is unset ────────────────────────────────────────
export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKROOT}/uv_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORKROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${WORKROOT}/apptainer_tmp}"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# ── Pins ───────────────────────────────────────────────────────────────────
# Python packages are pinned in setup/stack/pyproject.toml + uv.lock; slime and
# sglang resolve from the polar forks at fixed commits (setup/stack/repin.sh
# moves the pins). The slime checkout below is the same commit as the locked
# package: run.sh needs its train.py/train_async.py, convert_weights.sh its
# tools/, and Megatron its docker/patch/latest/megatron.patch. Megatron is the
# one checkout still installed as an editable over the lock (it carries slime's
# patch).
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="${SLIME_REPO:-$(sed -n 's/^slime = { git = "\([^"]*\)".*/\1/p' "${SCRIPT_DIR}/setup/stack/pyproject.toml")}"
SLIME_REF="${SLIME_REF:-$(sed -n 's/^slime = { git = "[^"]*", rev = "\([0-9a-f]*\)".*/\1/p' "${SCRIPT_DIR}/setup/stack/pyproject.toml")}"
[ -n "${SLIME_REPO}" ] && [ -n "${SLIME_REF}" ] || { echo "ERROR: could not read the slime git pin from setup/stack/pyproject.toml" >&2; exit 1; }
MEGATRON_DIR="${MEGATRON_DIR:-${WORKROOT}/Megatron-LM-slime-${SLIME_REF:0:12}}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-1dcf0dafa884ad52ffb243625717a3471643e087}"
MEGATRON_PATCH="${MEGATRON_PATCH:-${SLIME_DIR}/docker/patch/latest/megatron.patch}"

# ── Run placement (from the run config via launch.sh) ──────────────────────
: "${RUN_NAME:?pipeline.sh must be started by launch.sh}"
export RUN_ID="${RUN_ID:-${RUN_NAME}}"
export RUN_DIR="${WORKROOT}/harbor_slime_grpo/${RUN_ID}"
ASSET_DIR="${RUN_DIR}/assets"
mkdir -p "${ASSET_DIR}"
# Per-run copy of the setup environment (PATH, CUDA, caches). Jobs sharing a
# WORKROOT run concurrently; nothing job-written may live at WORKROOT level.
export ENV_FILE="${ENV_FILE:-${RUN_DIR}/env.sh}"
export HARNESS_DIR="${HARNESS_DIR_CFG:-${WORKROOT}/harbor_harness}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${WORKROOT}/harbor_sif_images}"
export HARBOR_DATASET_DIR="${TASKS_MOUNT_ROOT:-${TASKS_DIR}}"
export TORCH_DIST_DIR="${TORCH_DIST_DIR_CFG:-${TORCH_DIST_DIR:-${WORKROOT}/checkpoints/${HF_CHECKPOINT##*/}_torch_dist}}"
export REF_LOAD="${TORCH_DIST_DIR}"
export SAVE_ROOT="${SAVE_ROOT:-${WORKROOT}/ckpt/harbor_slime_grpo}"

INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-1}"
PREPARE_IMAGES="${PREPARE_IMAGES:-1}"
PREPARE_HARNESS="${PREPARE_HARNESS:-1}"
APPTAINER_PREPARE_JOBS="${APPTAINER_PREPARE_JOBS:-4}"
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"
RUN_TRAINING="${RUN_TRAINING:-1}"

# shellcheck source=./setup/common.sh
source "${SCRIPT_DIR}/setup/common.sh"

# Shared installs that are not atomic on their own (apptainer, checkouts, SIF
# images, harness CLI, weight conversion) are serialized across concurrent jobs
# in this WORKROOT; the later job finds them "present; skipping".
setup_lock_acquire "${WORKROOT}/.setup.lock"

if [ "${DRY_RUN}" = 0 ]; then
    # ── Environment ────────────────────────────────────────────────────────
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

    # ── External checkouts ─────────────────────────────────────────────────
    clone_retry() {   # clone_retry NAME REPO REF DEST — REF may be a branch/tag or a commit sha
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
    # The checkout must be the commit the lock installs (run.sh runs its train.py,
    # convert_weights.sh its tools/, Megatron its megatron.patch). After a repin the
    # existing checkout is moved to the pinned commit; local edits would be lost, so
    # a dirty tree is an error instead.
    slime_head="$(git -C "${SLIME_DIR}" rev-parse HEAD)"
    if [ "${slime_head}" != "${SLIME_REF}" ]; then
        [ -z "$(git -C "${SLIME_DIR}" status --porcelain)" ] || die "slime checkout ${SLIME_DIR} is at ${slime_head} with local changes; the lock pins ${SLIME_REF}"
        info "slime checkout at ${slime_head:0:12}; moving to the pinned ${SLIME_REF:0:12}"
        git -C "${SLIME_DIR}" remote set-url origin "${SLIME_REPO}" \
          && git -C "${SLIME_DIR}" fetch -q --depth 1 origin "${SLIME_REF}" && git -C "${SLIME_DIR}" checkout -q "${SLIME_REF}" \
          || die "could not check out slime ${SLIME_REF} in ${SLIME_DIR}"
    fi
    clone_retry Megatron-LM "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"
    if [ -f "${MEGATRON_PATCH}" ]; then
        if git -C "${MEGATRON_DIR}" apply --reverse --check "${MEGATRON_PATCH}" >/dev/null 2>&1; then
            info "slime megatron.patch already applied"
        else
            git -C "${MEGATRON_DIR}" apply --3way "${MEGATRON_PATCH}" && info "applied slime megatron.patch"
        fi
    fi
    export SLIME_DIR MEGATRON_DIR

    # ── Megatron editable ──────────────────────────────────────────────────
    # Megatron-LM is not in the lock (it carries slime's megatron.patch); it is
    # installed --no-deps as an editable of the patched checkout. Skipped when
    # the venv already points at this checkout.
    if [ "${INSTALL_EDITABLE}" = 1 ]; then
        log "Megatron-LM editable"
        overlay_current() {   # overlay_current DIST_NAME CHECKOUT_DIR
            "${PYTHON_BIN}" - "$1" "$2" <<'PYCHK'
import json, sys
from importlib.metadata import distribution, PackageNotFoundError
name, path = sys.argv[1], sys.argv[2].rstrip("/")
try:
    d = distribution(name)
    url = json.loads(d.read_text("direct_url.json") or "{}")
except (PackageNotFoundError, ValueError):
    sys.exit(1)
sys.exit(0 if url.get("dir_info", {}).get("editable") and url.get("url", "").rstrip("/").endswith(path) else 1)
PYCHK
        }
        if overlay_current megatron-core "${MEGATRON_DIR}"; then
            info "already installed from ${MEGATRON_DIR}; skipping"
        else
            uv pip install --python "${PYTHON_BIN}" --no-deps -e "${MEGATRON_DIR}"
        fi
    fi
    if [ "${INSTALL_TRAINING_STACK}" = 1 ] && [ "${SETUP_ENV}" = 1 ]; then
        # shellcheck source=./setup/ensure_training_stack.sh
        source "${SCRIPT_DIR}/setup/ensure_training_stack.sh"
    fi
else
    # Dry run: any python with pyyaml (the venv if it exists).
    PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
    [ -x "${PYTHON_BIN}" ] && "${PYTHON_BIN}" -c 'import yaml' 2>/dev/null || PYTHON_BIN="$(command -v python3)"
fi

# ── Tasks: materialize (tasks.dataset) → prompts + image list ──────────────
if [ -n "${TASKS_DATASET:-}" ] && [ ! -f "${TASKS_DIR}/manifest.json" ]; then
    log "dataset ${TASKS_DATASET} → ${TASKS_DIR}"
    [ "${DRY_RUN}" = 0 ] || die "dry run: tasks.dir does not exist yet; run without --dry-run once to materialize it"
    # shellcheck disable=SC2086
    "${PYTHON_BIN}" "${SCRIPT_DIR}/../datasets/${TASKS_DATASET}.py" --output "${TASKS_DIR}" ${TASKS_DATASET_ARGS}
fi
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
        log "harness ${HARNESS}${HARNESS_CLI_VERSION:+ @ ${HARNESS_CLI_VERSION}}"
        # harness.cli_version pins the CLI prepare_harness.sh installs.
        if [ -n "${HARNESS_CLI_VERSION:-}" ]; then
            case "${HARNESS}" in
                codex) export CODEX_VERSION="${HARNESS_CLI_VERSION}" ;;  opencode) export OPENCODE_VERSION="${HARNESS_CLI_VERSION}" ;;
                claude_code) export CLAUDE_CODE_VERSION="${HARNESS_CLI_VERSION}" ;;  qwen_code) export QWEN_CODE_VERSION="${HARNESS_CLI_VERSION}" ;;
                pi) export PI_VERSION="${HARNESS_CLI_VERSION}" ;;  mini_swe_agent) export MINI_SWE_AGENT_VERSION="${HARNESS_CLI_VERSION}" ;;  hermes) export HERMES_VERSION="${HARNESS_CLI_VERSION}" ;;
            esac
        fi
        bash "${SCRIPT_DIR}/prepare_harness.sh" "${HARNESS_DIR}" "${HARNESS}"
    fi

    # ── Checkpoint ─────────────────────────────────────────────────────────
    case "${HF_CHECKPOINT}" in
        /*|./*|../*|~*) ;;
        *)  log "HF snapshot ${HF_CHECKPOINT}"
            # mbridge reads *.safetensors from the local cache and does not download.
            "${PYTHON_BIN_DIR}/hf" download "${HF_CHECKPOINT}" >/dev/null && info "present in ${HF_HOME}" ;;
    esac
    if [ "${CONVERT_WEIGHTS}" = 1 ] || { [ "${CONVERT_WEIGHTS}" = auto ] && [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; }; then
        log "HF → torch_dist conversion"
        bash "${SCRIPT_DIR}/convert_weights.sh"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then
        "${PYTHON_BIN}" -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)' 2>/dev/null || true
    fi
fi

# ── Polar templates: @TOKENS@ from the run config, ${VARS} later by run.sh ──
setup_lock_release
log "polar templates"
# Judge-backed tasks: the key named by judge.api_key_env must be set on the host,
# otherwise the MCP/judge never starts and every reward is silently 0.
if [ -n "${RUBRIC_MODEL_API_KEY_ENV:-}" ]; then
    [ -n "${!RUBRIC_MODEL_API_KEY_ENV:-}" ] || die "judge.api_key_env=${RUBRIC_MODEL_API_KEY_ENV} is not set in the environment"
    info "judge key ${RUBRIC_MODEL_API_KEY_ENV}: set"
fi
export POLAR_CONFIG_TEMPLATE="${ASSET_DIR}/polar_config.yaml"
export TOPOLOGY_TEMPLATE="${ASSET_DIR}/topology.yaml"
"${PYTHON_BIN}" - "${SCRIPT_DIR}" "${ASSET_DIR}" <<'PY'
import json, os, sys, yaml
tpl_dir, out_dir = sys.argv[1:3]
env = os.environ
tokens = {
    "@HARNESS@": env["HARNESS"], "@HARNESS_MODEL_NAME@": env["HARNESS_MODEL_NAME"],
    "@HARNESS_DIR@": env["HARNESS_DIR"], "@HARBOR_DATASET_DIR@": env["HARBOR_DATASET_DIR"],
    "@RUN_NAME@": env["RUN_NAME"], "@GROUP_ID_SCOPE@": env["GROUP_ID_SCOPE"],
    "@OVERLONG_POLICY@": env.get("OVERLONG_POLICY", "zero_reward_train"),
    "@PATH_PREPEND@": (env.get("HARNESS_PATH_PREPEND", "").rstrip(":") + ":") if env.get("HARNESS_PATH_PREPEND") else "",
    "@LD_LIBRARY_PATH@": env.get("HARNESS_LD_LIBRARY_PATH", ""),
    # Rubric judge: model/base from the config, the key from the host env var it names.
    "@RUBRIC_MODEL@": env.get("RUBRIC_MODEL", ""), "@RUBRIC_MODEL_API_BASE@": env.get("RUBRIC_MODEL_API_BASE", ""),
    "@RUBRIC_MODEL_API_KEY@": os.environ.get(env.get("RUBRIC_MODEL_API_KEY_ENV", "") or "__unset__", ""),
}
typed = {
    "@SESSION_TIMEOUT@": int(env["SESSION_TIMEOUT"]), "@REQUEST_TIMEOUT@": int(env["REQUEST_TIMEOUT"]),
    "@MAX_ASYNC_LEVEL@": int(env["MAX_ASYNC_LEVEL"]), "@EOT_TOKEN_ID@": int(env["EOT_TOKEN_ID"]),
    "@MAX_RUN_WORKERS@": int(env["MAX_RUN_WORKERS"]),
    "@TIMEOUT_REWARD_ZERO@": env["TIMEOUT_REWARD_ZERO"] == "1",
    "@ENABLE_THINKING@": {"1": True, "0": False}.get(env.get("ENABLE_THINKING", ""), None),
    "@DROP_ZERO_VARIANCE_GROUPS@": env["DROP_ZERO_VARIANCE_GROUPS"] == "1",
}
def fill(node):
    if isinstance(node, dict): return {k: fill(v) for k, v in node.items() if not (isinstance(v, str) and v in typed and typed[v] is None)}
    if isinstance(node, list): return [fill(v) for v in node]
    if isinstance(node, str):
        if node in typed: return typed[node]
        for k, v in tokens.items(): node = node.replace(k, v)
    return node
for name in ("polar_config.yaml", "topology.yaml"):
    doc = fill(yaml.safe_load(open(os.path.join(tpl_dir, name))))
    if name == "polar_config.yaml":
        doc["polar_task_template"]["agent"]["settings"] = json.loads(env.get("HARNESS_SETTINGS_JSON") or "{}")
    with open(os.path.join(out_dir, name), "w") as f:
        f.write(f"# rendered by pipeline.sh from {os.path.join(tpl_dir, name)} and {env['RUN_CONFIG_PATH']}\n")
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, width=200)
print(f"rendered polar_config.yaml + topology.yaml -> {out_dir}")
PY
chmod 600 "${POLAR_CONFIG_TEMPLATE}"
export PREPARE_IMAGES=0

cat <<INFO

Prompts:    $(wc -l < "${PROMPT_DATA}" | tr -d " ") from ${TASKS_DIR}; $(wc -l < "${ASSET_DIR}/images.txt" | tr -d " ") image(s)
Harness:    ${HARNESS} from ${HARNESS_DIR}
Templates:  ${POLAR_CONFIG_TEMPLATE}, ${TOPOLOGY_TEMPLATE}
Save:       ${SAVE_ROOT}/${RUN_ID}
INFO
if [ "${DRY_RUN}" = 1 ]; then
    echo "--- rendered polar_config.yaml ---"; cat "${POLAR_CONFIG_TEMPLATE}"; exit 0
fi
[ "${RUN_TRAINING}" = 1 ] || exit 0
log "run.sh"
exec bash "${SCRIPT_DIR}/run.sh"
