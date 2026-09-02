#!/usr/bin/env bash
# Single-entry launcher for the SWE-Gym Slime GRPO example.
#
#   bash examples/swegym_slime_grpo/launch_e2e.sh
#
# Order: preflight → python stack → CUDA user space → apptainer → slime /
# Megatron checkouts + patches → editable installs → training stack → data →
# task images + agent CLIs → HF snapshot → weight conversion → run.sh.
# Every step is idempotent; re-running resumes where the previous run stopped.
#
# Placement: WORKROOT (default: <repo>/tmp) holds everything this script
# creates — checkouts, caches, toolchains, checkpoints. Cache variables you
# already export (HF_HOME, UV_CACHE_DIR, APPTAINER_CACHEDIR, ...) are left
# alone; only unset ones are pointed under WORKROOT.
# SETUP_ENV=0 skips preflight and all setup/ scripts (bring your own venv with
# CUDA torch, TE, and apptainer; PYTHON_BIN must then point at it).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PROJECT_ROOT
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
export ENV_FILE="${ENV_FILE:-${WORKROOT}/env.sh}"
export SETUP_ENV="${SETUP_ENV:-1}"
export TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
mkdir -p "${WORKROOT}"

# ── Caches: fill only what is unset ────────────────────────────────────────
export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKROOT}/uv_cache}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${WORKROOT}/apptainer_cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${WORKROOT}/apptainer_tmp}"
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${APPTAINER_CACHEDIR}" "${APPTAINER_TMPDIR}"

# ── Pins ───────────────────────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="${SLIME_REPO:-https://github.com/THUDM/slime.git}"
SLIME_REF="${SLIME_REF:-v0.3.0}"
# Slime v0.3.0's canonical Megatron commit (from its docker/Dockerfile) plus
# the companion patch its images apply. The 26.04-alpha.rc1 tag drifted and no
# longer ships megatron.training.tokenizer, which slime imports.
MEGATRON_DIR="${MEGATRON_DIR:-${WORKROOT}/Megatron-LM-slime-${SLIME_REF}}"
MEGATRON_REPO="${MEGATRON_REPO:-https://github.com/NVIDIA/Megatron-LM.git}"
MEGATRON_REF="${MEGATRON_REF:-1dcf0dafa884ad52ffb243625717a3471643e087}"
MEGATRON_PATCH="${MEGATRON_PATCH:-${SLIME_DIR}/docker/patch/latest/megatron.patch}"
SWEGYM_PACKAGE_SPEC="${SWEGYM_PACKAGE_SPEC:-swegym @ git+https://github.com/SWE-Gym/SWE-Bench-Package.git@16dd480cce9b27bf111a362d280881c6def5d2a7}"
MBRIDGE_VERSION="${MBRIDGE_VERSION:-0.15.1}"  # HF<->Megatron weight bridge (slime conversion)

# ── Model / run ────────────────────────────────────────────────────────────
export HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
export MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-model_args.sh}"   # model_args_9b.sh for Qwen3.5-9B
export TORCH_DIST_DIR="${TORCH_DIST_DIR:-${REF_LOAD:-${WORKROOT}/checkpoints/${HF_CHECKPOINT##*/}_torch_dist}}"
export REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR}}"
export RUN_ID="${RUN_ID:-${WANDB_RUN_ID:-swegym-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}}"
export SAVE_ROOT="${SAVE_ROOT:-${WORKROOT}/ckpt/swegym_slime_grpo}"
export SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
export RUN_DIR="${RUN_DIR:-${WORKROOT}/swegym_slime_grpo}"
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${WORKROOT}/swegym_agent_cli/opt_node}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${WORKROOT}/swegym_apptainer_images}"

INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
INSTALL_TRAINING_STACK="${INSTALL_TRAINING_STACK:-1}"
APPLY_SGLANG_PATCH="${APPLY_SGLANG_PATCH:-1}"
PREPARE_IMAGES="${PREPARE_IMAGES:-1}"
APPTAINER_PREPARE_JOBS="${APPTAINER_PREPARE_JOBS:-2}"
CONVERT_WEIGHTS="${CONVERT_WEIGHTS:-auto}"
RUN_TRAINING="${RUN_TRAINING:-1}"

# shellcheck source=./setup/common.sh
source "${SCRIPT_DIR}/setup/common.sh"

# ── Environment ────────────────────────────────────────────────────────────
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

# ── External checkouts ─────────────────────────────────────────────────────
clone_retry() {   # clone_retry NAME REPO REF DEST — REF may be a branch/tag or a commit sha
    local name="$1" repo="$2" ref="$3" dest="$4" i
    if [ -d "${dest}/.git" ]; then
        info "${name} checkout exists: ${dest}"; return 0
    fi
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

# ── Editable installs ──────────────────────────────────────────────────────
if [ "${INSTALL_EDITABLE}" = 1 ]; then
    log "editable installs"
    # [swebench] supplies the dependency tree the swegym fork (installed next) omits.
    "${UV_PIP[@]}" -e ".[swebench]"
    "${UV_PIP[@]}" -e "${SLIME_DIR}"
    "${UV_PIP[@]}" -e "${MEGATRON_DIR}"
    "${UV_PIP[@]}" --no-deps "mbridge==${MBRIDGE_VERSION}"
    if ! "${PYTHON_BIN}" -c 'from swegym.harness.constants import MAP_REPO_VERSION_TO_SPECS as m; assert {"dask/dask","python/mypy","pandas-dev/pandas"} <= set(m)' 2>/dev/null; then
        "${UV_PIP[@]}" "${SWEGYM_PACKAGE_SPEC}"
    fi
fi

if [ "${INSTALL_TRAINING_STACK}" = 1 ] && [ "${SETUP_ENV}" = 1 ]; then
    # shellcheck source=./setup/ensure_training_stack.sh
    source "${SCRIPT_DIR}/setup/ensure_training_stack.sh"
fi

if [ "${APPLY_SGLANG_PATCH}" = 1 ]; then
    log "sglang token-metadata patch"
    VIRTUAL_ENV="$(dirname "${PYTHON_BIN_DIR}")" bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_0513_token_metadata.sh"
fi

# ── Data, images, agent CLIs ───────────────────────────────────────────────
log "train data"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py"
if [ "${PREPARE_IMAGES}" = 1 ]; then
    log "task images + agent CLIs"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_apptainer_images.py" \
        --agent-cli-dir "${AGENT_CLI_DIR}" --image-dir "${APPTAINER_IMAGE_DIR}" \
        --cache-dir "${APPTAINER_CACHEDIR}" --tmp-dir "${APPTAINER_TMPDIR}" --jobs "${APPTAINER_PREPARE_JOBS}"
fi

# ── Checkpoint ─────────────────────────────────────────────────────────────
case "${HF_CHECKPOINT}" in
    /*|./*|../*|~*) ;;
    *)  log "HF snapshot ${HF_CHECKPOINT}"
        # The Megatron conversion (mbridge) reads *.safetensors from the local
        # cache and does not download them; make sure the full snapshot is present.
        "${PYTHON_BIN_DIR}/hf" download "${HF_CHECKPOINT}" >/dev/null && info "present in ${HF_HOME}" ;;
esac
if [ "${CONVERT_WEIGHTS}" = 1 ] || { [ "${CONVERT_WEIGHTS}" = auto ] && [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; }; then
    log "HF → torch_dist conversion"
    bash "${SCRIPT_DIR}/convert_weights.sh"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
    "${PYTHON_BIN}" -c 'import os, wandb; wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)' 2>/dev/null || true
fi

# ── Train ──────────────────────────────────────────────────────────────────
if [ "${RUN_TRAINING}" = 1 ]; then
    log "run.sh"
    exec bash "${SCRIPT_DIR}/run.sh"
fi
