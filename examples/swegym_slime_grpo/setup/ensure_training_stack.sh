#!/usr/bin/env bash
# Install the GPU training extras the editable installs do not pull, matching
# the CUDA major of the installed torch:
#   - Transformer Engine (Megatron needs it on every GPU; its torch bindings
#     build from source). torch cu13 → TE 2.14.0[core-cu13]; torch cu12 →
#     TE 2.5.0 (the upstream pin, which ships a cu12 core only).
#   - Flash Linear Attention (Qwen3.5 GatedDeltaNet layers).
#   - flash-attn 2.x from source on SM100/B200 only (TE's cuDNN backend has no
#     head_dim=256 kernel there).
# Idempotent: each component is skipped when importable. Ends with a hard gate.
# Sourced by launch_e2e.sh after ensure_cuda_userspace.sh and the editable installs.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SETUP_DIR}/common.sh"

TE_VERSION_CU13="${TE_VERSION_CU13:-2.14.0}"
TE_VERSION_CU12="${TE_VERSION_CU12:-2.5.0}"
FLASH_LINEAR_ATTENTION_VERSION="${FLASH_LINEAR_ATTENTION_VERSION:-0.5.0}"
# Source builds: cap parallelism (an uncapped TE/flash-attn build can exhaust
# node memory on shared machines).
BUILD_MAX_JOBS="${BUILD_MAX_JOBS:-8}"

cuda_major="$(torch_cuda_major)"
[ -n "${cuda_major}" ] || die "installed torch is not a CUDA build"
case "${cuda_major}" in
    13) te_spec="transformer-engine[core-cu13,pytorch]==${TE_VERSION_CU13}"; te_version="${TE_VERSION_CU13}" ;;
    12) te_spec="transformer-engine[pytorch]==${TE_VERSION_CU12}";           te_version="${TE_VERSION_CU12}" ;;
    *)  die "unsupported torch CUDA major ${cuda_major}" ;;
esac

te_ok() {
    [ "$(pip_version transformer-engine)" = "${te_version}" ] \
    && "${PYTHON_BIN}" -c 'import torch; torch.cuda.init(); import transformer_engine.pytorch' >/dev/null 2>&1
}

log "training stack: Transformer Engine ${te_version} (torch cu${cuda_major})"
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
    uv pip uninstall --python "${PYTHON_BIN}" transformer-engine transformer-engine-cu12 transformer-engine-cu13 transformer-engine-torch >/dev/null 2>&1 || true
    uv pip install --python "${PYTHON_BIN}" ninja pybind11 setuptools wheel >/dev/null
    info "building TE for sm_${arch} with MAX_JOBS=${BUILD_MAX_JOBS} (this takes a while)"
    CUDNN_PATH="${cudnn}" CUDNN_HOME="${cudnn}" NCCL_HOME="${nccl}" NCCL_INCLUDE_DIR="${nccl}/include" \
    CPATH="${nccl}/include:${cudnn}/include:${CUDA_HOME}/include:${CPATH:-}" \
    LIBRARY_PATH="${nccl}/lib:${cudnn}/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="${nccl}/lib:${cudnn}/lib:${LD_LIBRARY_PATH:-}" \
    NVTE_CUDA_ARCHS="${arch}" MAX_JOBS="${BUILD_MAX_JOBS}" NVTE_BUILD_THREADS_PER_JOB=1 \
        uv pip install --python "${PYTHON_BIN}" --torch-backend="${TORCH_BACKEND:-cu130}" --prerelease=allow \
            --no-build-isolation "${te_spec}"
fi
if [ "${cuda_major}" = 13 ]; then
    # Enforced by UV_OVERRIDE (setup/constraints.txt); apply to an existing venv too.
    want="$(grep -E '^nvidia-cublas==' "${SETUP_DIR}/constraints.txt" | cut -d= -f3)"
    [ "$(pip_version nvidia-cublas)" = "${want}" ] || uv pip install --python "${PYTHON_BIN}" "nvidia-cublas==${want}"
fi

log "training stack: Flash Linear Attention ${FLASH_LINEAR_ATTENTION_VERSION}"
if "${PYTHON_BIN}" -c "from fla.modules import FusedRMSNormGated, ShortConvolution; from fla.ops.gated_delta_rule import chunk_gated_delta_rule" >/dev/null 2>&1; then
    info "present; skipping"
else
    uv pip install --python "${PYTHON_BIN}" "flash-linear-attention==${FLASH_LINEAR_ATTENTION_VERSION}"
fi

if [ "${COMPUTE_CAP:-}" = "10.0" ]; then
    log "training stack: flash-attn 2.7.4.post1 (SM100 head_dim=256 fallback)"
    if "${PYTHON_BIN}" -c 'import flash_attn_2_cuda, flash_attn.flash_attn_interface' >/dev/null 2>&1; then
        info "present; skipping"
    else
        TORCH_CUDA_ARCH_LIST=10.0 FLASH_ATTN_CUDA_ARCHS=100 FLASH_ATTENTION_FORCE_BUILD=TRUE \
        MAX_JOBS="${BUILD_MAX_JOBS}" NVCC_THREADS=4 \
            uv pip install --python "${PYTHON_BIN}" --no-build-isolation flash-attn==2.7.4.post1
    fi
fi

log "training stack: gate"
te_ok || die "Transformer Engine ${te_version} is not importable after install"
"${PYTHON_BIN}" -c 'from importlib.metadata import version; print("  OK: transformer-engine", version("transformer-engine"))'
