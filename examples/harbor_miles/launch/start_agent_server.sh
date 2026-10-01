#!/usr/bin/env bash
# Start this node's Harbor agent server (singularity sandboxes on THIS node) in the
# background and register it for the Miles-side dispatcher.
#
#   start_agent_server.sh <run_dir> [port] [max_concurrent]
#
# Appends "http://<node-ip>:<port> <max_concurrent>" to $HARBOR_AGENT_SERVERS_FILE
# (default <run_dir>/agent_servers.txt) once /health answers. Logs: <run_dir>/agent_servers/<host>.log; trials under
# <run_dir>/trials/<host>/. Sandbox overlays live on node-local disk.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=../setup/common.sh
source "${HERE}/../setup/common.sh"
RUN_DIR="${1:?run_dir}"; PORT="$(hm_free_port "${2:-${HM_AGENT_SERVER_PORT:-65500}}")"; MAXC="${3:-${HM_SANDBOXES_PER_NODE:-32}}"
HOST="$(hostname -s)"
NODE_IP="$(hm_node_ip)"

HARBOR_DIR="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_harbor.sh")"
SERVER_PY="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_sandbox_runtime.sh")"

export HARBOR_ENV_TYPE=singularity
export HARBOR_TASKS_DIR="${HARBOR_TASKS_DIR:?HARBOR_TASKS_DIR (task packages root) must be set}"
export HARBOR_SINGULARITY_BIN="${HM_APPTAINER}/bin/singularity"
export HARBOR_SINGULARITY_SERVER_PYTHON="${SERVER_PY}"
export HARBOR_SINGULARITY_RUNTIME_DIR="$(cd "$(dirname "${SERVER_PY}")/.." && pwd)"
export HARBOR_SINGULARITY_SIF_DIRS="${HM_SIF_DIRS}"
export HARBOR_SINGULARITY_CACHE_DIR="${HM_ROOT}/sif_cache"
export HARBOR_SINGULARITY_START_TIMEOUT_SEC="${HARBOR_SINGULARITY_START_TIMEOUT_SEC:-600}"
# Per-trial writable overlays on node-local disk (--writable-tmpfs is 64 MB): the first
# writable of HM_OVERLAY_BASE, /local, /tmp.
for base in ${HM_OVERLAY_BASE:-} /local /tmp; do
    if mkdir -p "${base}/hm-overlay-${USER}-${SLURM_JOB_ID:-local}" 2>/dev/null; then
        export HARBOR_SINGULARITY_OVERLAY_DIR="${base}/hm-overlay-${USER}-${SLURM_JOB_ID:-local}"
        break
    fi
done
[ -n "${HARBOR_SINGULARITY_OVERLAY_DIR:-}" ] || hm_die "no writable node-local dir for sandbox overlays"
hm_log "sandbox overlays under ${HARBOR_SINGULARITY_OVERLAY_DIR} ($(df -h "${HARBOR_SINGULARITY_OVERLAY_DIR}" | awk 'NR==2{print $4}') free)"
mkdir -p "${HARBOR_SINGULARITY_CACHE_DIR}"
# Read-only agent toolchains (setup/ensure_agent_tools.sh), bound into every sandbox at the same path.
MOUNTS=""
for tool in ${HM_AGENT_TOOLS:-mini-swe-agent}; do
    root="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_agent_tools.sh" "${tool}")"
    var="HM_AGENT_TOOLS_$(echo "${tool}" | tr 'a-z-' 'A-Z_')"
    export "${var}=${root}"
    MOUNTS="${MOUNTS:+${MOUNTS},}{\"source\": \"${root}\", \"target\": \"${root}\", \"read_only\": true}"
    hm_log "agent tools ${tool}: ${root} ($(cat "${root}/VERSION" 2>/dev/null))"
done
export HARBOR_SINGULARITY_MOUNTS="[${MOUNTS}]"
export PYTHONPATH="${HM_EXAMPLE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

mkdir -p "${RUN_DIR}/agent_servers" "${RUN_DIR}/trials/${HOST}"
LOG="${RUN_DIR}/agent_servers/${HOST}.log"
hm_log "agent server on ${NODE_IP}:${PORT} (max ${MAXC} sandboxes), harbor ${HARBOR_DIR}"
(
    cd "${HARBOR_DIR}"
    exec "${HARBOR_DIR}/.venv/bin/python" miles_agent_server.py \
        --host 0.0.0.0 --port "${PORT}" --max-concurrent "${MAXC}" \
        --trials-dir "${RUN_DIR}/trials/${HOST}" --dashboard-port 0 \
        ${HM_AGENT_TIMEOUT:+--agent-timeout "${HM_AGENT_TIMEOUT}"} \
        --dashboard-log-path "${RUN_DIR}/agent_servers/${HOST}.requests.jsonl"
) >"${LOG}" 2>&1 &
echo $! > "${RUN_DIR}/agent_servers/${HOST}.pid"
for _ in $(seq 1 180); do
    if curl -fs "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "http://${NODE_IP}:${PORT} ${MAXC}" >> "${HARBOR_AGENT_SERVERS_FILE:-${RUN_DIR}/agent_servers.txt}"
        hm_log "agent server ready: http://${NODE_IP}:${PORT}"
        exit 0
    fi
    sleep 2
done
tail -50 "${LOG}" >&2
hm_die "agent server did not become healthy"
