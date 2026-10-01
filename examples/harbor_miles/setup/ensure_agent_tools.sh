#!/usr/bin/env bash
# Build (once per version) agent toolchains that sandboxes use read-only, instead of installing
# them inside every sandbox at trial time (per-trial apt + uv + pip over the network: slow at
# dozens of concurrent starts, fails where sandboxes have no DNS (dfw), and lets versions drift).
#   ${HM_ROOT}/agent-tools/<name>-<ver>/   python (relocatable, uv-managed) + uv tool env + bin/
# The directory is bind-mounted into sandboxes at the same absolute path (start_agent_server.sh).
# Prints the toolchain root on stdout.
#   ensure_agent_tools.sh mini-swe-agent [version]
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${HERE}/common.sh"
NAME="${1:?agent name}"; VERSION="${2:-${HM_MSWEA_VERSION:-}}"
case "${NAME}" in
    mini-swe-agent) ;;
    *) hm_die "ensure_agent_tools.sh: no recipe for ${NAME}" ;;
esac
HARBOR_DIR="$(bash "${HERE}/ensure_harbor.sh")"
TAG="${NAME}-${VERSION:-latest-$(date +%Y%m%d)}"
DEST="${HM_ROOT}/agent-tools/${TAG}"

build_tools() {
    if [ -e "${DEST}" ]; then
        mkdir -p "${HM_ROOT}/to_delete"; mv "${DEST}" "${HM_ROOT}/to_delete/${TAG}.partial-$(date +%s)"
    fi
    mkdir -p "${DEST}"
    export UV_PYTHON_INSTALL_DIR="${DEST}/python" UV_TOOL_DIR="${DEST}/tools" UV_TOOL_BIN_DIR="${DEST}/bin"
    "${HM_UV}" python install 3.12 >&2
    spec="mini-swe-agent${VERSION:+==${VERSION}}"
    "${HM_UV}" tool install --python 3.12 "${spec}" --with 'litellm[proxy]' >&2
    tool_py="${DEST}/tools/mini-swe-agent/bin/python"
    # The same install-time patches Harbor applies inside the sandbox (output cap, step timing).
    "${HARBOR_DIR}/.venv/bin/python" - "${tool_py}" <<'PY' >&2
import subprocess, sys
from harbor.agents.installed import mini_swe_agent as m
py = sys.argv[1]
for name, src in (("output cap", m._OUTPUT_CAP_PATCH.format(cap_bytes=m.MAX_COMMAND_OUTPUT_BYTES)),
                  ("step timing", m._STEP_TIMING_PATCH)):
    r = subprocess.run([py, "-"], input=src, text=True, capture_output=True)
    print(f"{name}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()[-300:]}")
    if r.returncode:
        sys.exit(1)
PY
    "${DEST}/bin/mini-swe-agent" --help >/dev/null
    "${tool_py}" -c "import importlib.metadata as m; print(m.version('mini-swe-agent'))" > "${DEST}/VERSION"
    hm_log "agent tools: ${NAME} $(cat "${DEST}/VERSION") at ${DEST}"
}
hm_once "agent-tools-${TAG}" build_tools >&2
echo "${DEST}"
