#!/usr/bin/env bash
# Build (once per version) agent toolchains that sandboxes use read-only, instead of installing
# them inside every sandbox at trial time (per-trial apt + uv + pip over the network: slow at
# dozens of concurrent starts, fails where sandboxes have no DNS (dfw), and lets versions drift).
#   ${HM_ROOT}/agent-tools/<name>-<ver>/   python (relocatable, uv-managed) + uv tool env + bin/
# The directory is bind-mounted into sandboxes at the same absolute path (start_agent_server.sh).
# Prints the toolchain root on stdout.
#   ensure_agent_tools.sh mini-swe-agent [version]     (default 2.4.6)
#   ensure_agent_tools.sh codex [version]              (default 0.125.0; Node 22 + @openai/codex)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${HERE}/common.sh"
NAME="${1:?agent name}"
case "${NAME}" in
    mini-swe-agent) VERSION="${2:-${HM_MSWEA_VERSION:-2.4.6}}" ;;
    codex) VERSION="${2:-${HM_CODEX_VERSION:-0.125.0}}" ;;
    *) hm_die "ensure_agent_tools.sh: no recipe for ${NAME}" ;;
esac
NODE_VERSION="${HM_NODE_VERSION:-22.20.0}"
HARBOR_DIR="$(bash "${HERE}/ensure_harbor.sh")"
TAG="${NAME}-${VERSION}"
DEST="${HM_ROOT}/agent-tools/${TAG}"

build_codex() {
    set -e
    if [ -e "${DEST}" ]; then
        mkdir -p "${HM_ROOT}/to_delete"; mv "${DEST}" "${HM_ROOT}/to_delete/${TAG}.partial-$(date +%s)"
    fi
    mkdir -p "${DEST}"
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o "${DEST}/node.tar.xz" || return 1
    mkdir -p "${DEST}/node" && tar -xJf "${DEST}/node.tar.xz" -C "${DEST}/node" --strip-components=1 || return 1
    rm -f "${DEST}/node.tar.xz"
    export PATH="${DEST}/node/bin:${PATH}" npm_config_cache="${DEST}/npm-cache"
    npm install -g --prefix "${DEST}/node" "@openai/codex@${VERSION}" >&2 || return 1
    mkdir -p "${DEST}/bin"
    ln -sfn ../node/bin/codex "${DEST}/bin/codex"; ln -sfn ../node/bin/node "${DEST}/bin/node"
    HOME="${DEST}/build-home" "${DEST}/bin/codex" --version > "${DEST}/VERSION" || return 1
    hm_log "agent tools: codex $(cat "${DEST}/VERSION") at ${DEST}"
}

build_tools() {
    set -e   # not inherited: this runs inside hm_once's `cmd && touch stamp`
    if [ -e "${DEST}" ]; then
        mkdir -p "${HM_ROOT}/to_delete"; mv "${DEST}" "${HM_ROOT}/to_delete/${TAG}.partial-$(date +%s)"
    fi
    mkdir -p "${DEST}"
    export UV_PYTHON_INSTALL_DIR="${DEST}/python" UV_TOOL_DIR="${DEST}/tools" UV_TOOL_BIN_DIR="${DEST}/bin"
    export UV_PYTHON_BIN_DIR="${DEST}/python-bin"   # not ~/.local/bin ($HOME is full/read-only on some clusters)
    "${HM_UV}" python install 3.12 >&2
    "${HM_UV}" tool install --python 3.12 "mini-swe-agent==${VERSION}" --with 'litellm[proxy]' >&2 || return 1
    tool_py="${DEST}/tools/mini-swe-agent/bin/python"
    # The same install-time patches Harbor applies inside the sandbox (output cap, step timing).
    # minisweagent creates its config dir under $HOME on import (full/read-only on some clusters).
    mkdir -p "${DEST}/build-home"
    HOME="${DEST}/build-home" "${HARBOR_DIR}/.venv/bin/python" - "${tool_py}" <<'PY' >&2 || return 1
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
    # minisweagent prints a banner on import: take the last line
    local_py="$(HOME="${DEST}/build-home" "${tool_py}" -c 'import minisweagent.environments.local as m; print(m.__file__)' | tail -1)"
    grep -q "harbor: elided" "${local_py}" || { hm_log "output-cap patch missing in ${local_py}"; return 1; }
    HOME="${DEST}/build-home" "${DEST}/bin/mini-swe-agent" --help >/dev/null || return 1
    "${tool_py}" -c "import importlib.metadata as m; print(m.version('mini-swe-agent'))" > "${DEST}/VERSION" || return 1
    hm_log "agent tools: ${NAME} $(cat "${DEST}/VERSION") at ${DEST}"
}
case "${NAME}" in
    codex) hm_once "agent-tools-${TAG}" build_codex >&2 ;;
    *) hm_once "agent-tools-${TAG}" build_tools >&2 ;;
esac
echo "${DEST}"
