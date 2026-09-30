#!/usr/bin/env bash
# Build (once per pin+patch set) the Harbor checkout + venv the agent servers run from:
#   ${HM_ROOT}/harbor/<pin12>-<patch12>/  (git checkout at harbor/PIN with harbor/patches/*.patch
#                                          applied, uv venv in .venv)
# Prints the checkout path on stdout. Idempotent and safe to run on every node at once.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${HERE}/common.sh"

PIN="$(tr -d '[:space:]' < "${HM_EXAMPLE_DIR}/harbor/PIN")"
PATCH_HASH="$(cat "${HM_EXAMPLE_DIR}"/harbor/patches/*.patch | sha256sum | cut -c1-12)"
DEST="${HM_ROOT}/harbor/${PIN:0:12}-${PATCH_HASH}"

build_harbor() {
    if [ -e "${DEST}" ]; then   # a partial earlier attempt: set aside, never delete
        mkdir -p "${HM_ROOT}/to_delete"
        mv "${DEST}" "${HM_ROOT}/to_delete/$(basename "${DEST}").partial-$(date +%s)"
    fi
    hm_log "harbor: clone ${PIN} -> ${DEST}"
    mkdir -p "${DEST}"
    git -C "${DEST}" init -q
    git -C "${DEST}" remote add origin https://github.com/harbor-framework/harbor.git
    git -C "${DEST}" fetch -q --depth 1 origin "${PIN}"
    git -C "${DEST}" checkout -q --detach FETCH_HEAD
    for p in "${HM_EXAMPLE_DIR}"/harbor/patches/*.patch; do
        hm_log "harbor: apply $(basename "${p}")"
        git -C "${DEST}" apply --whitespace=nowarn "${p}"
    done
    hm_log "harbor: uv sync"
    (cd "${DEST}" && UV_PROJECT_ENVIRONMENT="${DEST}/.venv" "${HM_UV}" sync --frozen --no-dev --python 3.12 >&2)
    "${DEST}/.venv/bin/python" -c "import harbor, fastapi, uvicorn; from harbor.environments.singularity.singularity import find_prebuilt_sif" >&2
    hm_log "harbor: ready at ${DEST}"
}
hm_once "harbor-${PIN:0:12}-${PATCH_HASH}" build_harbor >&2
echo "${DEST}"
