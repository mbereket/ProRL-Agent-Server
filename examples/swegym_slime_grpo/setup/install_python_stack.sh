#!/usr/bin/env bash
# Create the venv and install Polar + SGLang with a consistent CUDA torch build.
#
# Why this is not just `uv pip install -e . sglang==0.5.13`:
#   - The repo's [tool.uv] torch-backend="auto" selects the torch build from the
#     *driver*. On a driver older than CUDA 13 that yields a cpu or cu12x torch
#     while sglang's CUDA-13-only pins (cuda-python 13.x, flash-attn-4) are
#     marker-gated away: the resolve "succeeds" into a broken set whose first
#     symptom is `operator torchvision::nms does not exist` at import.
#   - cu126/cu128/cu129 cannot resolve against those pins at all; cu130 resolves
#     only with --prerelease=allow (flash-attn-4 has pre-release versions only).
#   - The whole torch family must share one local-version tag (+cu130).
# Sourced by launch_e2e.sh after preflight (needs NEED_UV, WORKROOT, PROJECT_ROOT).
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SETUP_DIR}/common.sh"

TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
SGLANG_SPEC="${SGLANG_SPEC:-sglang==0.5.13}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_FAMILY=(torch torchvision torchaudio torchao torchcodec)

log "python stack: uv"
if [ "${NEED_UV:-0}" = 1 ]; then
    mkdir -p "${WORKROOT}/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${WORKROOT}/bin" UV_NO_MODIFY_PATH=1 sh
    prepend_path PATH "${WORKROOT}/bin"
fi
info "$(uv --version)"
# Every uv call below (and in launch_e2e.sh) honors these overrides.
emit_export UV_OVERRIDE "${SETUP_DIR}/constraints.txt"

log "python stack: venv ${VENV_DIR} (python ${PYTHON_VERSION})"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi
PYTHON_BIN="${VENV_DIR}/bin/python"
emit_export PYTHON_BIN "${PYTHON_BIN}"
emit_export VIRTUAL_ENV "${VENV_DIR}"
prepend_path PATH "${VENV_DIR}/bin"

log "python stack: polar + ${SGLANG_SPEC} with torch ${TORCH_BACKEND}"
(cd "${PROJECT_ROOT}" && uv pip install --python "${PYTHON_BIN}" \
    --torch-backend="${TORCH_BACKEND}" --prerelease=allow -e . "${SGLANG_SPEC}")

# Verify the torch family shares the requested local-version tag; repair if not
# (a pre-existing venv may carry a mixed set).
mismatched=()
for p in "${TORCH_FAMILY[@]}"; do
    v="$(pip_version "$p")"
    [ -n "$v" ] && [[ "$v" != *"+${TORCH_BACKEND}"* ]] && mismatched+=("$p=$v")
done
if [ "${#mismatched[@]}" -gt 0 ]; then
    info "torch family mismatch (${mismatched[*]}); reinstalling together"
    reinstall=(); for p in "${TORCH_FAMILY[@]}"; do reinstall+=(--reinstall-package "$p"); done
    (cd "${PROJECT_ROOT}" && uv pip install --python "${PYTHON_BIN}" \
        --torch-backend="${TORCH_BACKEND}" --prerelease=allow -e . "${SGLANG_SPEC}" "${reinstall[@]}")
fi
for p in "${TORCH_FAMILY[@]}" numpy scipy wandb nvidia-cublas flash-attn-4 cuda-python sglang; do
    v="$(pip_version "$p")"; [ -n "$v" ] && info "$(printf '%-14s %s' "$p" "$v")"
done
"${PYTHON_BIN}" -c 'import torch; print("  torch.version.cuda", torch.version.cuda)'
