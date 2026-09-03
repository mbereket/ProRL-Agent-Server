#!/usr/bin/env bash
# Build the GPU training extras the lock cannot ship as wheels:
#   - Transformer Engine torch bindings (transformer-engine-torch), built from
#     source against the locked torch and the locked TE core
#     (transformer-engine-cu13). Its runtime and build deps are in the lock, so
#     it is installed --no-deps.
#   - flash-attn 2.x from source on SM100/B200 only (TE's cuDNN backend has no
#     head_dim=256 kernel there).
# Idempotent: each component is skipped when importable. Ends with a hard gate.
# Sourced by pipeline.sh after ensure_cuda_userspace.sh and the editable overlay.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SETUP_DIR}/common.sh"

# Source builds: cap parallelism (an uncapped TE/flash-attn build can exhaust
# node memory on shared machines).
BUILD_MAX_JOBS="${BUILD_MAX_JOBS:-8}"

te_version="$(pip_version transformer-engine-cu13)"
[ -n "${te_version}" ] || die "transformer-engine-cu13 is not installed; run install_python_stack.sh first"
[ "$(torch_cuda_major)" = 13 ] || die "installed torch is not a CUDA 13 build"

te_ok() {
    [ "$(pip_version transformer-engine-torch)" = "${te_version}" ] \
    && "${PYTHON_BIN}" -c 'import torch; torch.cuda.init(); import transformer_engine.pytorch' >/dev/null 2>&1
}

log "training stack: Transformer Engine torch bindings ${te_version}"
if te_ok; then
    info "present; skipping"
else
    [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ] || die "CUDA_HOME with nvcc is required to build TE (${CUDA_HOME:-unset})"
    cudnn="$("${PYTHON_BIN}" -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])' 2>/dev/null || true)"
    [ -n "${cudnn}" ] || die "pip nvidia-cudnn not found in the venv; is torch a CUDA build?"
    nccl_h="$(find "${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}" -path '*/nvidia/nccl/include/nccl.h' -print -quit)"
    [ -n "${nccl_h}" ] || die "pip nvidia-nccl headers not found in the venv"
    nccl="$(dirname "$(dirname "${nccl_h}")")"
    arch="${NVTE_CUDA_ARCHS:-${COMPUTE_CAP//./}}"
    # A pre-existing venv may carry a cu12 core or stale bindings; the locked
    # metapackage + cu13 core stay (TE's import check needs both, same version).
    uv pip uninstall --python "${PYTHON_BIN}" transformer-engine-cu12 transformer-engine-torch >/dev/null 2>&1 || true
    info "building for sm_${arch} with MAX_JOBS=${BUILD_MAX_JOBS} (this takes a while)"
    CUDNN_PATH="${cudnn}" CUDNN_HOME="${cudnn}" NCCL_HOME="${nccl}" NCCL_INCLUDE_DIR="${nccl}/include" \
    CPATH="${nccl}/include:${cudnn}/include:${CUDA_HOME}/include:${CPATH:-}" \
    LIBRARY_PATH="${nccl}/lib:${cudnn}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="${nccl}/lib:${cudnn}/lib:${LD_LIBRARY_PATH:-}" \
    NVTE_CUDA_ARCHS="${arch}" MAX_JOBS="${BUILD_MAX_JOBS}" NVTE_BUILD_THREADS_PER_JOB=1 \
        uv pip install --python "${PYTHON_BIN}" --no-build-isolation --no-deps --reinstall \
            "transformer-engine-torch==${te_version}"
fi

if [ "${COMPUTE_CAP:-}" = "10.0" ]; then
    log "training stack: flash-attn 2.7.4.post1 (SM100 head_dim=256 fallback)"
    if "${PYTHON_BIN}" -c 'import flash_attn_2_cuda, flash_attn.flash_attn_interface' >/dev/null 2>&1; then
        info "present; skipping"
    else
        TORCH_CUDA_ARCH_LIST=10.0 FLASH_ATTN_CUDA_ARCHS=100 FLASH_ATTENTION_FORCE_BUILD=TRUE \
        MAX_JOBS="${BUILD_MAX_JOBS}" NVCC_THREADS=4 \
            uv pip install --python "${PYTHON_BIN}" --no-build-isolation --no-deps flash-attn==2.7.4.post1
    fi
fi

log "training stack: gate"
if ! te_ok; then
    # Show the actual import error before dying (te_ok swallows it).
    echo "  transformer-engine-torch dist version: $(pip_version transformer-engine-torch)"
    "${PYTHON_BIN}" -c 'import torch; torch.cuda.init(); import transformer_engine.pytorch' 2>&1 | tail -15 || true
    die "Transformer Engine ${te_version} is not importable after install"
fi
"${PYTHON_BIN}" -c 'from importlib.metadata import version; print("  OK: transformer-engine-torch", version("transformer-engine-torch"))'
