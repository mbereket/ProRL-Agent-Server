#!/usr/bin/env bash
# Host-side python for the Polar services (rollout server, gateways), config
# rendering and task preparation. Polar runs on the host, not in the Miles image:
# its gateways start the task sandboxes with the host's Apptainer. The venv holds
# Polar's dependencies only; Polar itself is imported from ${PROJECT_ROOT}/src via
# PYTHONPATH, so one venv serves every checkout. Sourced by launch.sh (needs
# WORKROOT, PROJECT_ROOT, ENV_FILE); exports POLAR_PYTHON.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=../../../harbor_slime_grpo/internal/setup/common.sh
source "${PROJECT_ROOT}/examples/harbor_slime_grpo/internal/setup/common.sh"

POLAR_VENV="${POLAR_VENV:-${WORKROOT}/polar_venv}"
# Bump when the dependency set below changes.
POLAR_VENV_STAMP="polar-venv-v1 fastapi uvicorn httpx pydantic pyyaml transformers jinja2 huggingface_hub"

log "polar venv ${POLAR_VENV}"
if ! command -v uv >/dev/null 2>&1; then
    if [ ! -x "${WORKROOT}/bin/uv" ]; then
        mkdir -p "${WORKROOT}/bin"
        curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${WORKROOT}/bin" UV_NO_MODIFY_PATH=1 sh
    fi
    prepend_path PATH "${WORKROOT}/bin"
fi
if [ "$(cat "${POLAR_VENV}/.stamp" 2>/dev/null)" != "${POLAR_VENV_STAMP}" ]; then
    uv venv -q --allow-existing --python 3.12 "${POLAR_VENV}"
    # shellcheck disable=SC2086
    uv pip install -q --python "${POLAR_VENV}/bin/python" ${POLAR_VENV_STAMP#* }
    echo "${POLAR_VENV_STAMP}" > "${POLAR_VENV}/.stamp"
fi
emit_export POLAR_PYTHON "${POLAR_VENV}/bin/python"
PYTHONPATH="${PROJECT_ROOT}/src" "${POLAR_PYTHON}" -c 'import polar, yaml, transformers; print("  polar", polar.__file__)'
