#!/usr/bin/env bash
# Preflight: gather machine facts, decide which environment fixes are needed,
# and fail fast with a specific message for anything this example cannot fix.
#
# Standalone:  bash examples/harbor_slime_grpo/setup/preflight.sh
# From launch_e2e.sh it is sourced so the decisions become variables:
#   NEED_CUDA_COMPAT   driver too old for the CUDA-13 torch build this example pins
#   NEED_CUDA_TOOLKIT  no CUDA 13 nvcc (Transformer Engine builds from source)
#   NEED_APPTAINER     no apptainer/singularity binary; install unprivileged
#   NEED_UV            uv missing; bootstrap into WORKROOT/bin
#   NEED_PIXI          pixi missing (only relevant with NEED_CUDA_TOOLKIT); bootstrap into WORKROOT/pixi
#
# Nothing here installs or modifies anything.
set -euo pipefail

SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SETUP_DIR}/common.sh"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SETUP_DIR}/../../.." && pwd)}"
WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"

# The pinned stack (sglang 0.5.13 → cuda-python 13.x, flash-attn-4) is CUDA-13
# only; torch is installed from the cu130 index. See setup/constraints.txt.
REQUIRED_CUDA_MAJOR="${REQUIRED_CUDA_MAJOR:-13}"
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"

POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-8080}"
POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-8100}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"

FATAL=()
fatal() { FATAL+=("$*"); }
row()   { printf '  %-22s %s\n' "$1" "$2"; }

log "preflight: machine facts"
row host "$(hostname)"

# ── GPUs / driver ─────────────────────────────────────────────────────────
GPU_COUNT=0; GPU_NAME=""; DRIVER_VERSION=""; DRIVER_CUDA_MAX=""; COMPUTE_CAP=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
    read -r GPU_NAME DRIVER_VERSION COMPUTE_CAP < <(
        nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null \
        | head -1 | awk -F', ' '{gsub(/ /,"_",$1); print $1, $2, $3}')
    # "CUDA Version" in the nvidia-smi banner is the newest CUDA runtime the
    # kernel driver supports natively (without forward-compat libraries).
    DRIVER_CUDA_MAX="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | awk '{print $3}' | head -1)"
fi
row gpus "${GPU_COUNT} × ${GPU_NAME:-none}"
row driver "${DRIVER_VERSION:-none} (native CUDA ≤ ${DRIVER_CUDA_MAX:-?}, compute cap ${COMPUTE_CAP:-?})"
if [ "${GPU_COUNT}" -lt 1 ]; then
    fatal "no NVIDIA GPUs visible (nvidia-smi). This example needs 8 GPUs per node."
fi

NEED_CUDA_COMPAT=0
if [ -n "${DRIVER_CUDA_MAX}" ] && [ "${DRIVER_CUDA_MAX%%.*}" -lt "${REQUIRED_CUDA_MAJOR}" ]; then
    NEED_CUDA_COMPAT=1
    row "cuda compat" "NEEDED: driver supports CUDA ${DRIVER_CUDA_MAX}, torch ${TORCH_BACKEND} needs ${REQUIRED_CUDA_MAJOR}.x → forward-compat libs"
else
    row "cuda compat" "not needed"
fi

# ── CUDA toolkit (nvcc) ───────────────────────────────────────────────────
NEED_CUDA_TOOLKIT=1; NVCC_BIN=""; NVCC_VERSION=""
for cand in "${CUDA_HOME:+${CUDA_HOME}/bin/nvcc}" "$(command -v nvcc 2>/dev/null || true)" /usr/local/cuda/bin/nvcc; do
    [ -n "${cand}" ] && [ -x "${cand}" ] || continue
    NVCC_VERSION="$("${cand}" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
    if [ "${NVCC_VERSION%%.*}" = "${REQUIRED_CUDA_MAJOR}" ]; then
        NVCC_BIN="${cand}"; NEED_CUDA_TOOLKIT=0; break
    fi
done
if [ "${NEED_CUDA_TOOLKIT}" = 1 ]; then
    row nvcc "NEEDED: no CUDA ${REQUIRED_CUDA_MAJOR}.x nvcc found${NVCC_VERSION:+ (found ${NVCC_VERSION})} → conda-forge toolkit via pixi"
    # The toolkit install runs through pixi; find it or plan to bootstrap it.
    if [ -x "${PIXI_HOME:-${WORKROOT}/pixi}/bin/pixi" ]; then
        row pixi "${PIXI_HOME:-${WORKROOT}/pixi}/bin/pixi"
    elif command -v pixi >/dev/null 2>&1; then
        row pixi "$(command -v pixi)"
    else
        NEED_PIXI=1
        row pixi "NEEDED: not found → bootstrap into ${PIXI_HOME:-${WORKROOT}/pixi}"
    fi
else
    row nvcc "${NVCC_BIN} (${NVCC_VERSION})"
fi
NEED_PIXI="${NEED_PIXI:-0}"

# ── Compiler (TE's torch bindings always build from source) ───────────────
for c in gcc g++; do
    if command -v "$c" >/dev/null 2>&1; then
        row "$c" "$("$c" --version | head -1)"
    else
        fatal "$c not found; Transformer Engine builds its torch bindings from source."
    fi
done

# ── Container runtime ─────────────────────────────────────────────────────
NEED_APPTAINER=0; APPTAINER_FOUND=""
if [ -n "${POLAR_APPTAINER_BIN:-}" ] && [ -x "${POLAR_APPTAINER_BIN}" ]; then
    APPTAINER_FOUND="${POLAR_APPTAINER_BIN}"
else
    APPTAINER_FOUND="$(command -v apptainer 2>/dev/null || command -v singularity 2>/dev/null || true)"
fi
if [ -n "${APPTAINER_FOUND}" ]; then
    row apptainer "${APPTAINER_FOUND} ($("${APPTAINER_FOUND}" --version 2>/dev/null))"
else
    NEED_APPTAINER=1
    row apptainer "NEEDED: none on PATH → unprivileged install into WORKROOT"
    # The unprivileged installer unpacks RPMs: needs cpio and rpm2cpio (or busybox to shim it).
    command -v cpio >/dev/null 2>&1 || fatal "cpio not found; the unprivileged Apptainer install needs it (or set POLAR_APPTAINER_BIN)."
    if ! command -v rpm2cpio >/dev/null 2>&1 && ! command -v busybox >/dev/null 2>&1; then
        fatal "neither rpm2cpio nor busybox found; cannot unpack Apptainer RPMs (or set POLAR_APPTAINER_BIN)."
    fi
fi
# Unprivileged apptainer (and a non-setuid system install) need user namespaces.
if unshare -U true 2>/dev/null; then
    row "user namespaces" "available"
else
    row "user namespaces" "NOT available"
    if [ "${NEED_APPTAINER}" = 1 ]; then
        fatal "no apptainer/singularity and unprivileged user namespaces are disabled; an unprivileged install cannot work here. Ask your admins for apptainer or set POLAR_APPTAINER_BIN."
    fi
fi

# ── Tools ─────────────────────────────────────────────────────────────────
for c in git curl tar xz envsubst; do
    command -v "$c" >/dev/null 2>&1 || fatal "$c not found on PATH."
done
NEED_UV=0
if command -v uv >/dev/null 2>&1; then
    row uv "$(uv --version)"
else
    NEED_UV=1; row uv "NEEDED: not on PATH → bootstrap into WORKROOT/bin"
fi

# ── Filesystems ───────────────────────────────────────────────────────────
mkdir -p "${WORKROOT}" 2>/dev/null || fatal "cannot create WORKROOT=${WORKROOT}"
if touch "${WORKROOT}/.preflight-write-test" 2>/dev/null; then
    rm -f "${WORKROOT}/.preflight-write-test"
    row WORKROOT "${WORKROOT} ($(df -h "${WORKROOT}" | awk 'NR==2 {print $4}') free)"
else
    fatal "WORKROOT=${WORKROOT} is not writable."
fi
if touch "${HOME}/.preflight-write-test" 2>/dev/null; then
    rm -f "${HOME}/.preflight-write-test"; HOME_WRITABLE=1
    row HOME "${HOME} ($(df -h "${HOME}" | awk 'NR==2 {print $4}') free)"
else
    HOME_WRITABLE=0
    row HOME "${HOME} NOT writable (caches will be placed under WORKROOT)"
fi

# ── Ports ─────────────────────────────────────────────────────────────────
port_in_use() {
    if command -v ss >/dev/null 2>&1; then ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$";
    else (echo > "/dev/tcp/127.0.0.1/$1") 2>/dev/null; fi
}
for spec in "POLAR_ROLLOUT_PORT:${POLAR_ROLLOUT_PORT}" "POLAR_GATEWAY_PORT:${POLAR_GATEWAY_PORT}" \
            "SGLANG_ROUTER_PORT:${SGLANG_ROUTER_PORT}" "RAY_DASHBOARD_PORT:${RAY_DASHBOARD_PORT}" "RAY_GCS_PORT:${RAY_GCS_PORT}"; do
    name="${spec%%:*}"; port="${spec##*:}"
    if port_in_use "${port}"; then
        fatal "port ${port} (${name}) is already in use on this host; export ${name}=<free port> and re-run."
    fi
done
row ports "free: rollout ${POLAR_ROLLOUT_PORT}, gateway ${POLAR_GATEWAY_PORT}, router ${SGLANG_ROUTER_PORT}, ray ${RAY_DASHBOARD_PORT}/${RAY_GCS_PORT}"

# ── Network ───────────────────────────────────────────────────────────────
reach() { curl -sI --max-time 8 "$1" >/dev/null 2>&1; }
for spec in "https://github.com|cloning slime and Megatron-LM" \
            "https://pypi.org|python packages" \
            "https://huggingface.co|model checkpoint and datasets" \
            "https://download.pytorch.org|cu130 torch wheels"; do
    url="${spec%%|*}"; why="${spec##*|}"
    if reach "${url}"; then row "${url#https://}" reachable; else fatal "${url} unreachable (needed for ${why})."; fi
done
if [ "${NEED_CUDA_COMPAT}" = 1 ] && ! reach https://developer.download.nvidia.com; then
    fatal "developer.download.nvidia.com unreachable (needed for cuda-compat libraries)."
fi
if [ "${NEED_PIXI}" = 1 ] && ! reach https://pixi.sh; then
    fatal "pixi.sh unreachable (needed to bootstrap pixi for the CUDA toolkit)."
fi
if [ "${NEED_CUDA_TOOLKIT}" = 1 ] && ! reach https://conda.anaconda.org; then
    fatal "conda.anaconda.org unreachable (needed for the conda-forge CUDA toolkit)."
fi
if [ "${NEED_APPTAINER}" = 1 ] && ! reach https://raw.githubusercontent.com; then
    fatal "raw.githubusercontent.com unreachable (needed for the Apptainer installer)."
fi
if [ "${NEED_UV}" = 1 ] && ! reach https://astral.sh; then
    fatal "astral.sh unreachable (needed to bootstrap uv)."
fi

# ── Verdict ───────────────────────────────────────────────────────────────
log "preflight: decisions"
row NEED_CUDA_COMPAT "${NEED_CUDA_COMPAT}"
row NEED_CUDA_TOOLKIT "${NEED_CUDA_TOOLKIT}"
row NEED_APPTAINER "${NEED_APPTAINER}"
row NEED_UV "${NEED_UV}"
row NEED_PIXI "${NEED_PIXI}"
if [ "${#FATAL[@]}" -gt 0 ]; then
    printf '\npreflight FAILED (%d problem(s) this example cannot fix):\n' "${#FATAL[@]}" >&2
    for f in "${FATAL[@]}"; do printf '  - %s\n' "$f" >&2; done
    exit 1
fi
export NEED_CUDA_COMPAT NEED_CUDA_TOOLKIT NEED_APPTAINER NEED_UV NEED_PIXI HOME_WRITABLE COMPUTE_CAP GPU_COUNT NVCC_BIN
echo "preflight OK"
