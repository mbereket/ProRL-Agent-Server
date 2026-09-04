#!/usr/bin/env bash
# Create the venv and install the locked python stack (internal/setup/stack/uv.lock).
#
# The lock pins one consistent set — SGLang 0.5.13 and Slime v0.3.0 from the
# polar forks (git sources at fixed commits), their dependency trees, torch
# 2.11+cu130, Transformer Engine 2.12 cu13 core, Polar (editable) — so the
# install is a single `uv sync --frozen`. What the lock cannot carry is layered
# on afterwards: the patched Megatron checkout (pipeline.sh, editable) and the
# Transformer Engine torch bindings source build (ensure_training_stack.sh).
# The sync is --inexact so those layered packages survive a re-run. uv holds
# its own lock on the environment while syncing, and a sync against a venv
# that already matches the lock changes nothing.
# Sourced by pipeline.sh after preflight (needs NEED_UV, WORKROOT, PROJECT_ROOT).
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SETUP_DIR}/common.sh"

STACK_DIR="${SETUP_DIR}/stack"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"

log "python stack: uv"
if [ "${NEED_UV:-0}" = 1 ]; then
    mkdir -p "${WORKROOT}/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${WORKROOT}/bin" UV_NO_MODIFY_PATH=1 sh
    prepend_path PATH "${WORKROOT}/bin"
fi
info "$(uv --version)"

log "python stack: uv sync --frozen (${STACK_DIR}/uv.lock) → ${VENV_DIR}"
# UV_PROJECT_ENVIRONMENT places the project venv; python version comes from
# stack/.python-version (uv downloads it if the machine has none).
UV_PROJECT_ENVIRONMENT="${VENV_DIR}" uv sync --frozen --inexact --project "${STACK_DIR}"
PYTHON_BIN="${VENV_DIR}/bin/python"
emit_export PYTHON_BIN "${PYTHON_BIN}"
emit_export VIRTUAL_ENV "${VENV_DIR}"
prepend_path PATH "${VENV_DIR}/bin"

for p in torch torchvision torchaudio numpy scipy wandb nvidia-cublas flash-attn-4 cuda-python sglang transformer-engine-cu13; do
    v="$(pip_version "$p")"; [ -n "$v" ] && info "$(printf '%-24s %s' "$p" "$v")"
done
"${PYTHON_BIN}" -c 'import torch; print("  torch.version.cuda", torch.version.cuda)'
