# shellcheck shell=bash
# Shared helpers for the setup/ scripts. Source, do not execute.
#
# Contract: pipeline.sh exports PROJECT_ROOT, WORKROOT, ENV_FILE and
# PYTHON_BIN before sourcing any setup/ script. Scripts append `export` lines to
# ENV_FILE so run.sh, convert_weights.sh and multinode workers see the same
# CUDA/apptainer environment as the setup shell.

log()  { printf '\n=== %s ===\n' "$*"; }
info() { printf '  %s\n' "$*"; }
die()  { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# emit_export VAR VALUE — export now and persist to ENV_FILE (idempotent per VAR).
emit_export() {
    local var="$1" value="$2"
    export "${var}=${value}"
    mkdir -p "$(dirname "${ENV_FILE}")"
    touch "${ENV_FILE}"
    if grep -q "^export ${var}=" "${ENV_FILE}" 2>/dev/null; then
        local tmp; tmp="$(mktemp)"
        grep -v "^export ${var}=" "${ENV_FILE}" > "${tmp}" || true; mv "${tmp}" "${ENV_FILE}"
    fi
    printf 'export %s=%q\n' "${var}" "${value}" >> "${ENV_FILE}"
}

# prepend_path VAR DIR — prepend DIR to a colon-separated path variable, persisted.
prepend_path() {
    local var="$1" dir="$2" cur="${!1:-}"
    case ":${cur}:" in *":${dir}:"*) return ;; esac
    emit_export "${var}" "${dir}${cur:+:${cur}}"
}

# fetch URL DEST — curl with retries; fails loudly.
fetch() {
    local url="$1" dest="$2"
    curl -fL --retry 5 --retry-delay 10 --connect-timeout 20 -o "${dest}.part" "${url}" \
        || die "download failed: ${url}"
    mv "${dest}.part" "${dest}"
}

# pip_version NAME — installed version in the active venv, or empty.
pip_version() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
try:
    print(version(sys.argv[1]))
except PackageNotFoundError:
    pass
PY
}

# torch_cuda_major — CUDA major of the installed torch build (e.g. 13), or empty.
torch_cuda_major() {
    "${PYTHON_BIN}" - <<'PY' 2>/dev/null
import torch
v = torch.version.cuda
print(v.split(".")[0] if v else "")
PY
}

# config_python SCRIPT ARGS... — run a python script that needs pyyaml, before
# the venv necessarily exists: the project venv, else uv with pyyaml, else a
# system python that has yaml.
config_python() {
    local venv_py="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
    if [ -x "${venv_py}" ] && "${venv_py}" -c 'import yaml' 2>/dev/null; then
        "${venv_py}" "$@"
    elif command -v uv >/dev/null 2>&1; then
        uv run -q --no-project --with pyyaml --python 3.12 python "$@"
    elif command -v python3 >/dev/null 2>&1 && python3 -c 'import yaml' 2>/dev/null; then
        python3 "$@"
    else
        die "no python with pyyaml found (install uv: https://astral.sh/uv, or pyyaml for python3)"
    fi
}

# setup_lock_acquire FILE / setup_lock_release — serialize shared setup across
# concurrent jobs in one WORKROOT (flock on fd 9; waits up to 2 h). If the
# filesystem lacks flock support we warn and continue unlocked.
setup_lock_acquire() {
    local lock="$1"
    mkdir -p "$(dirname "${lock}")"
    exec 9>>"${lock}"
    if ! flock -w 7200 9 2>/dev/null; then
        echo "  WARNING: could not take setup lock ${lock} (no flock support or 2 h timeout); continuing unlocked" >&2
    fi
}
setup_lock_release() { flock -u 9 2>/dev/null || true; exec 9>&-; }
