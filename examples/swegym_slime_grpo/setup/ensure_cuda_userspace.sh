#!/usr/bin/env bash
# Make the CUDA-13 torch build runnable and buildable on this machine.
#
#   NEED_CUDA_COMPAT=1  → NVIDIA cuda-compat forward-compatibility libraries
#                         (user-space CUDA 13 driver on an older kernel driver;
#                         supported on data-center GPUs). Prepended to
#                         LD_LIBRARY_PATH; nothing system-wide changes.
#   NEED_CUDA_TOOLKIT=1 → conda-forge CUDA toolkit (nvcc, NVTX headers) in a
#                         pixi environment under WORKROOT, for the Transformer
#                         Engine source build.
# Both are pinned to 13.1: TE 2.14's cu13 core needs a ≥13.1 cuBLASLt and the
# cu130 torch runs on any 13.x user space (minor-version compatibility).
# Always ends with a hard gate: torch must see the GPU and run a matmul.
# Sourced by launch_e2e.sh after install_python_stack.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

CUDA_COMPAT_VERSION="${CUDA_COMPAT_VERSION:-590.48.01_cuda13.1}"
CUDA_COMPAT_URL="${CUDA_COMPAT_URL:-https://developer.download.nvidia.com/compute/cuda/redist/cuda_compat/linux-x86_64/cuda_compat-linux-x86_64-${CUDA_COMPAT_VERSION}-archive.tar.xz}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-13.1}"

if [ "${NEED_CUDA_COMPAT:-0}" = 1 ]; then
    log "cuda: forward-compat libraries ${CUDA_COMPAT_VERSION}"
    root="${WORKROOT}/cuda-compat-${CUDA_COMPAT_VERSION}"
    if [ -z "$(find "${root}" -name 'libcuda.so.1' 2>/dev/null)" ]; then
        mkdir -p "${root}"
        fetch "${CUDA_COMPAT_URL}" "${root}.tar.xz"
        tar -xJf "${root}.tar.xz" -C "${root}" --strip-components=1
    fi
    libdir="$(dirname "$(find "${root}" -name 'libcuda.so.1' | head -n1)")"
    prepend_path LD_LIBRARY_PATH "${libdir}"
    info "libcuda.so.1 from ${libdir}"
fi

if [ "${NEED_CUDA_TOOLKIT:-0}" = 1 ]; then
    log "cuda: conda-forge toolkit ${CUDA_TOOLKIT_VERSION} via pixi"
    emit_export PIXI_HOME "${PIXI_HOME:-${WORKROOT}/pixi}"
    emit_export PIXI_CACHE_DIR "${PIXI_CACHE_DIR:-${WORKROOT}/pixi-cache}"
    prepend_path PATH "${PIXI_HOME}/bin"
    if [ "${NEED_PIXI:-0}" = 1 ] || ! command -v pixi >/dev/null 2>&1; then
        curl -fsSL https://pixi.sh/install.sh | bash
    fi
    tk="${WORKROOT}/cuda-toolkit-${CUDA_TOOLKIT_VERSION}"
    if [ ! -f "${tk}/pixi.toml" ]; then
        pixi init "${tk}" -c conda-forge -p linux-64
        pixi add -m "${tk}/pixi.toml" "cuda=${CUDA_TOOLKIT_VERSION}.*" "cuda-nvtx-dev=${CUDA_TOOLKIT_VERSION}.*"
    else
        pixi install -m "${tk}/pixi.toml"
    fi
    cuda_home="${tk}/.pixi/envs/default"
    emit_export CUDA_HOME "${cuda_home}"
    prepend_path PATH "${cuda_home}/bin"
    prepend_path LIBRARY_PATH "${cuda_home}/lib"
    prepend_path LIBRARY_PATH "${cuda_home}/targets/x86_64-linux/lib"
    prepend_path CPATH "${cuda_home}/include"
    prepend_path CPATH "${cuda_home}/targets/x86_64-linux/include"
    # Toolkit libs go AFTER the compat libcuda (append, not prepend).
    emit_export LD_LIBRARY_PATH "${LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:}${cuda_home}/lib:${cuda_home}/targets/x86_64-linux/lib"
    info "$(nvcc --version | tail -1)"
elif [ -n "${NVCC_BIN:-}" ]; then
    emit_export CUDA_HOME "${CUDA_HOME:-$(dirname "$(dirname "${NVCC_BIN}")")}"
fi

log "cuda: gate (torch sees the GPU and runs a matmul)"
"${PYTHON_BIN}" - <<'PY' || die "CUDA gate failed. Driver/torch mismatch not resolved; see messages above."
import torch
assert torch.cuda.is_available(), (
    "torch.cuda.is_available() is False. torch build: cu%s" % torch.version.cuda)
x = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize(); (x @ x).sum().item()
print(f"  OK: torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY
