#!/usr/bin/env bash
# Build the agent-harness directory that is bind-mounted read-only into every
# task container. Task images then need nothing preinstalled: Node-based CLIs
# (codex, opencode, claude, qwen, pi) come from <dir>/node, Python-based ones
# (mini-swe-agent, hermes) from a uv tool install with its own managed interpreter.
#
#   prepare_harness.sh <harness_dir> [harness ...]
#
# The directory is mounted at the SAME absolute path inside the container
# (Python venvs carry absolute interpreter paths), so build it on the shared
# filesystem the compute nodes see. Idempotent: existing installs at the pinned
# version are kept. Needs network (login node).
set -euo pipefail
HARNESS_DIR="${1:?usage: prepare_harness.sh <harness_dir> [harness ...]}"
shift
HARNESSES=("$@")
[ "${#HARNESSES[@]}" -gt 0 ] || HARNESSES=(mini_swe_agent)

NODE_VERSION="${NODE_VERSION:-22.11.0}"
# Pins: codex must match polar.agent.presets.codex DEFAULT_CODEX_VERSION (the
# preset hard-fails on a version mismatch); the rest are the versions the
# tmax-15k rollout example uses.
CODEX_VERSION="${CODEX_VERSION:-0.125.0}"
OPENCODE_VERSION="${OPENCODE_VERSION:-1.4.6}"
CLAUDE_CODE_VERSION="${CLAUDE_CODE_VERSION:-2.1.111}"
QWEN_CODE_VERSION="${QWEN_CODE_VERSION:-0.14.5}"
PI_VERSION="${PI_VERSION:-0.67.68}"
MINI_SWE_AGENT_VERSION="${MINI_SWE_AGENT_VERSION:-2.4.2}"
HERMES_VERSION="${HERMES_VERSION:-0.15.1}"
HARNESS_PYTHON="${HARNESS_PYTHON:-3.12}"

mkdir -p "${HARNESS_DIR}/bin"
HARNESS_DIR="$(cd "${HARNESS_DIR}" && pwd)"
NODE_DIR="${HARNESS_DIR}/node"
export npm_config_cache="${npm_config_cache:-${HARNESS_DIR}/.npm-cache}"
export npm_config_update_notifier=false

need_node=0
for h in "${HARNESSES[@]}"; do
    case "$h" in codex|opencode|claude_code|qwen_code|pi) need_node=1 ;; esac
done
if [ "${need_node}" = 1 ] && [ ! -x "${NODE_DIR}/bin/node" ]; then
    echo "[harness] installing node ${NODE_VERSION} into ${NODE_DIR}"
    mkdir -p "${NODE_DIR}"
    curl -fL --retry 5 "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
        | tar -xJ -C "${NODE_DIR}" --strip-components=1
fi
npm_install() {   # npm_install <bin> <package@version>
    local bin="$1" pkg="$2" want="${2##*@}" have=""
    have="$(PATH="${NODE_DIR}/bin:${PATH}" "${NODE_DIR}/bin/${bin}" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)"
    if [ "${have}" = "${want}" ]; then echo "[harness] ${pkg} present"; return 0; fi
    echo "[harness] installing ${pkg}"
    PATH="${NODE_DIR}/bin:${PATH}" "${NODE_DIR}/bin/npm" install -g --no-audit --no-fund --prefix="${NODE_DIR}" "${pkg}"
    ln -sf "../node/bin/${bin}" "${HARNESS_DIR}/bin/${bin}"
}

for h in "${HARNESSES[@]}"; do
    case "$h" in
        codex)       npm_install codex "@openai/codex@${CODEX_VERSION}" ;;
        opencode)    npm_install opencode "opencode-ai@${OPENCODE_VERSION}" ;;
        claude_code) npm_install claude "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" ;;
        qwen_code)   npm_install qwen "@qwen-code/qwen-code@${QWEN_CODE_VERSION}" ;;
        pi)          npm_install pi "@mariozechner/pi-coding-agent@${PI_VERSION}" ;;
        mini_swe_agent)
            command -v uv >/dev/null || { echo "ERROR: uv is required to install mini-swe-agent" >&2; exit 1; }
            have="$("${HARNESS_DIR}/bin/mini-swe-agent" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 || true)"
            if [ "${have}" = "${MINI_SWE_AGENT_VERSION}" ]; then echo "[harness] mini-swe-agent ${have} present"; continue; fi
            echo "[harness] installing mini-swe-agent ${MINI_SWE_AGENT_VERSION} (python ${HARNESS_PYTHON})"
            # Everything (interpreter, tool venv, entry point) lives under HARNESS_DIR
            # so the mount at the same path resolves the venv's absolute shebangs.
            UV_PYTHON_INSTALL_DIR="${HARNESS_DIR}/uv-python" UV_TOOL_DIR="${HARNESS_DIR}/uv-tools" \
            UV_TOOL_BIN_DIR="${HARNESS_DIR}/bin" \
                uv tool install --force --python "${HARNESS_PYTHON}" --python-preference only-managed \
                    "mini-swe-agent==${MINI_SWE_AGENT_VERSION}"
            ;;
        hermes)
            command -v uv >/dev/null || { echo "ERROR: uv is required to install hermes-agent" >&2; exit 1; }
            if [ -x "${HARNESS_DIR}/bin/hermes" ] && [ -f "${HARNESS_DIR}/.hermes-${HERMES_VERSION}" ]; then echo "[harness] hermes ${HERMES_VERSION} present"; continue; fi
            echo "[harness] installing hermes-agent ${HERMES_VERSION} (python ${HARNESS_PYTHON})"
            UV_PYTHON_INSTALL_DIR="${HARNESS_DIR}/uv-python" UV_TOOL_DIR="${HARNESS_DIR}/uv-tools" \
            UV_TOOL_BIN_DIR="${HARNESS_DIR}/bin" \
                uv tool install --force --python "${HARNESS_PYTHON}" --python-preference only-managed \
                    "hermes-agent==${HERMES_VERSION}" && touch "${HARNESS_DIR}/.hermes-${HERMES_VERSION}"
            ;;
        *) echo "ERROR: unknown harness ${h} (codex|opencode|claude_code|qwen_code|pi|hermes|mini_swe_agent)" >&2; exit 1 ;;
    esac
done
[ -x "${NODE_DIR}/bin/node" ] && ln -sf ../node/bin/node "${HARNESS_DIR}/bin/node" && ln -sf ../node/bin/npm "${HARNESS_DIR}/bin/npm" || true
echo "[harness] ready: ${HARNESS_DIR} (bin: $(ls "${HARNESS_DIR}/bin" | tr '\n' ' '))"
