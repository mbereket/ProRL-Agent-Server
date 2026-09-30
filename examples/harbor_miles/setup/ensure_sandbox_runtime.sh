#!/usr/bin/env bash
# Build (once) the python that runs Harbor's exec server INSIDE every sandbox:
# a relocatable python-build-standalone CPython with fastapi+uvicorn in its own
# site-packages, bind-mounted read-only at the same path into each container.
# With it, sandbox start does no apt/pip/network work (the offline bootstrap of
# patch 0001). No PYTHONPATH/venv activation leaks into the task's processes.
# Prints the python path on stdout.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${HERE}/common.sh"

VERSION="py312-fastapi-v1"
DEST="${HM_ROOT}/sandbox-runtime/${VERSION}"

build_runtime() {
    if [ -e "${DEST}" ]; then
        mkdir -p "${HM_ROOT}/to_delete"
        mv "${DEST}" "${HM_ROOT}/to_delete/${VERSION}.partial-$(date +%s)"
    fi
    mkdir -p "${DEST}"
    hm_log "sandbox runtime: python 3.12 -> ${DEST}"
    "${HM_UV}" python install 3.12 --install-dir "${DEST}/python" >&2
    py="$(ls -d "${DEST}"/python/cpython-3.12*/bin/python3.12 | head -1)"
    [ -x "${py}" ] || hm_die "no python in ${DEST}/python"
    "${HM_UV}" pip install --python "${py}" --break-system-packages --no-cache \
        "fastapi==0.115.14" "uvicorn==0.34.3" "pydantic>=2.9,<3" >&2
    mkdir -p "${DEST}/bin"
    ln -sfn "${py}" "${DEST}/bin/python3"
    "${DEST}/bin/python3" -c "import fastapi, uvicorn, pydantic" >&2
}
hm_once "sandbox-runtime-${VERSION}" build_runtime >&2
echo "${DEST}/bin/python3"
