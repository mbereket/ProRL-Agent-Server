#!/usr/bin/env bash
# Build the agent-harness directory that is bind-mounted read-only into every
# task sandbox, so task images need nothing preinstalled.
#
#   prepare_harness.sh <harness_dir> <codex|opencode|mini_swe_agent> [version]
#
# codex and opencode are npm packages under <dir>/node (Node 22); mini-swe-agent
# is a uv tool install with its own interpreter. The directory is mounted at the
# SAME absolute path inside the container (Python venvs carry absolute paths), so
# build it on the shared filesystem. Idempotent: an install at the requested
# version is kept. Needs network.
set -euo pipefail
HARNESS_DIR="${1:?usage: prepare_harness.sh <harness_dir> <harness> [version]}"
HARNESS="${2:?usage: prepare_harness.sh <harness_dir> <harness> [version]}"
VERSION="${3:-}"
NODE_VERSION="${NODE_VERSION:-22.11.0}"
# codex must match polar.agent.presets.codex DEFAULT_CODEX_VERSION (the preset hard-fails on a mismatch).
case "${HARNESS}" in
    codex)          pkg="@openai/codex"; bin=codex; VERSION="${VERSION:-0.125.0}" ;;
    opencode)       pkg="opencode-ai";   bin=opencode; VERSION="${VERSION:-1.4.6}" ;;
    mini_swe_agent) pkg="mini-swe-agent"; bin=mini-swe-agent; VERSION="${VERSION:-2.4.2}" ;;
    *) echo "ERROR: unknown harness ${HARNESS} (codex|opencode|mini_swe_agent)" >&2; exit 1 ;;
esac

mkdir -p "${HARNESS_DIR}/bin"
HARNESS_DIR="$(cd "${HARNESS_DIR}" && pwd)"
have="$(PATH="${HARNESS_DIR}/node/bin:${PATH}" "${HARNESS_DIR}/bin/${bin}" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)"
if [ "${have}" = "${VERSION}" ]; then echo "[harness] ${pkg} ${VERSION} present in ${HARNESS_DIR}"; exit 0; fi

if [ "${HARNESS}" = mini_swe_agent ]; then
    echo "[harness] installing ${pkg}==${VERSION}"
    # Interpreter, tool venv and entry point all live under HARNESS_DIR so the
    # mount at the same path resolves the venv's absolute shebangs.
    UV_PYTHON_INSTALL_DIR="${HARNESS_DIR}/uv-python" UV_TOOL_DIR="${HARNESS_DIR}/uv-tools" UV_TOOL_BIN_DIR="${HARNESS_DIR}/bin" \
        uv tool install --force --python 3.12 --python-preference only-managed "${pkg}==${VERSION}"
else
    NODE_DIR="${HARNESS_DIR}/node"
    if [ ! -x "${NODE_DIR}/bin/node" ]; then
        echo "[harness] installing node ${NODE_VERSION}"
        mkdir -p "${NODE_DIR}"
        curl -fL --retry 5 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
            | tar -xJ -C "${NODE_DIR}" --strip-components=1
        ln -sf ../node/bin/node "${HARNESS_DIR}/bin/node"; ln -sf ../node/bin/npm "${HARNESS_DIR}/bin/npm"
    fi
    echo "[harness] installing ${pkg}@${VERSION}"
    export npm_config_cache="${npm_config_cache:-${HARNESS_DIR}/.npm-cache}" npm_config_update_notifier=false
    PATH="${NODE_DIR}/bin:${PATH}" "${NODE_DIR}/bin/npm" install -g --no-audit --no-fund --prefix="${NODE_DIR}" "${pkg}@${VERSION}"
    ln -sf "../node/bin/${bin}" "${HARNESS_DIR}/bin/${bin}"
fi
echo "[harness] ready: ${HARNESS_DIR}/bin/${bin} ($(PATH="${HARNESS_DIR}/node/bin:${PATH}" "${HARNESS_DIR}/bin/${bin}" --version 2>/dev/null | tail -1))"
