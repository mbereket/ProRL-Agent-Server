#!/usr/bin/env bash
# Slime and Megatron-LM checkouts. Sourced by launch.sh after the python stack.
#
# The lock (stack/uv.lock) installs slime as a package from the polar fork at a
# fixed commit; the checkout here must be that same commit because run.sh runs
# its train.py, convert_weights.sh its tools/, and Megatron takes its
# docker/patch/latest/megatron.patch. Megatron-LM is not in the lock (it carries
# slime's patch): cloned, patched, installed --no-deps as an editable.
# Exports SLIME_DIR, MEGATRON_DIR.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SETUP_DIR}/common.sh"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
SLIME_REPO="$(sed -n 's/^slime = { git = "\([^"]*\)".*/\1/p' "${SETUP_DIR}/stack/pyproject.toml")"
SLIME_REF="$(sed -n 's/^slime = { git = "[^"]*", rev = "\([0-9a-f]*\)".*/\1/p' "${SETUP_DIR}/stack/pyproject.toml")"
[ -n "${SLIME_REPO}" ] && [ -n "${SLIME_REF}" ] || die "could not read the slime git pin from setup/stack/pyproject.toml"
MEGATRON_DIR="${MEGATRON_DIR:-${WORKROOT}/Megatron-LM-slime-${SLIME_REF:0:12}}"
MEGATRON_REPO="https://github.com/NVIDIA/Megatron-LM.git"
MEGATRON_REF="1dcf0dafa884ad52ffb243625717a3471643e087"   # the commit slime v0.3.0's Dockerfile uses

clone_at() {   # clone_at NAME REPO SHA DEST — shallow checkout of one commit, 5 tries
    local name="$1" repo="$2" sha="$3" dest="$4" i
    if [ -d "${dest}/.git" ]; then info "${name} checkout exists: ${dest}"; return 0; fi
    [ -e "${dest}" ] && die "${name} path exists but is not a git checkout: ${dest}"
    for i in 1 2 3 4 5; do
        mkdir -p "${dest}" && git -C "${dest}" init -q && git -C "${dest}" remote add origin "${repo}" \
          && git -C "${dest}" fetch -q --depth 1 origin "${sha}" && git -C "${dest}" checkout -q FETCH_HEAD && return 0
        info "clone of ${name} failed (try ${i}/5); retrying in 20s"; rm -rf "${dest}"; sleep 20
    done
    die "could not clone ${name} from ${repo}"
}

log "checkouts"
clone_at slime "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
head="$(git -C "${SLIME_DIR}" rev-parse HEAD)"
if [ "${head}" != "${SLIME_REF}" ]; then
    # After a repin: a clean checkout moves to the pinned commit; one with local
    # changes is set aside (never deleted) and the pinned commit cloned fresh.
    if [ -n "$(git -C "${SLIME_DIR}" status --porcelain)" ]; then
        stale="${SLIME_DIR}.stale-${head:0:12}"
        info "slime checkout at ${head:0:12} has local changes; setting it aside as ${stale}"
        mv "${SLIME_DIR}" "${stale}"; clone_at slime "${SLIME_REPO}" "${SLIME_REF}" "${SLIME_DIR}"
    else
        info "slime checkout at ${head:0:12}; moving to the pinned ${SLIME_REF:0:12}"
        git -C "${SLIME_DIR}" remote set-url origin "${SLIME_REPO}"
        git -C "${SLIME_DIR}" fetch -q --depth 1 origin "${SLIME_REF}" && git -C "${SLIME_DIR}" checkout -q "${SLIME_REF}" \
          || die "could not check out slime ${SLIME_REF} in ${SLIME_DIR}"
    fi
fi
clone_at Megatron-LM "${MEGATRON_REPO}" "${MEGATRON_REF}" "${MEGATRON_DIR}"
patch="${SLIME_DIR}/docker/patch/latest/megatron.patch"
if git -C "${MEGATRON_DIR}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    info "slime megatron.patch already applied"
else
    git -C "${MEGATRON_DIR}" apply --3way "${patch}" && info "applied slime megatron.patch"
fi
export SLIME_DIR MEGATRON_DIR

log "Megatron-LM editable"
if "${PYTHON_BIN}" - "${MEGATRON_DIR}" <<'PY'
import json, sys
from importlib.metadata import distribution, PackageNotFoundError
try:
    url = json.loads(distribution("megatron-core").read_text("direct_url.json") or "{}")
except (PackageNotFoundError, ValueError):
    sys.exit(1)
sys.exit(0 if url.get("dir_info", {}).get("editable") and url.get("url", "").rstrip("/").endswith(sys.argv[1].rstrip("/")) else 1)
PY
then
    info "already installed from ${MEGATRON_DIR}"
else
    uv pip install --python "${PYTHON_BIN}" --no-deps -e "${MEGATRON_DIR}"
fi
