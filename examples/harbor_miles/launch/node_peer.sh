#!/usr/bin/env bash
# Harbor agent server + node monitor on a NON-head node of a multi-node job.
#
#   node_peer.sh <config.env>      (spawned by node_entry.sh on the head: one overlapping 1-task srun step per other node)
#
# Why: the slurm-compose step runs node_entry.sh on node 0 only, and miles_runtime/ray_node.sh starts only Ray workers on
# the other nodes, so without this the other nodes run no sandboxes. The step blocks while the agent server lives
# (sandboxes are its children), and the job's end tears it down.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
CFG="${1:?config.env}"
[ -f "${CFG}" ] || CFG="${HM_EXAMPLE_DIR}/${CFG}"
set -a; source "${CFG}"; set +a
export RUN_DIR="${HM_ROOT}/runs/${RUN_NAME:?}"
export HARBOR_AGENT_SERVERS_FILE="${RUN_DIR}/agent_servers-${SLURM_JOB_ID}.txt"
HM_AGENT_TIMEOUT="${HM_AGENT_TIMEOUT:-3600}" \
    bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${RUN_DIR}" "${HM_AGENT_SERVER_PORT:-65500}" \
    "${HM_SANDBOXES_PER_NODE:-32}"
if [ "${HM_NODE_MONITOR:-1}" = 1 ] && command -v python3 >/dev/null; then
    _me="$(grep -m1 "//$(hm_node_ip):" "${HARBOR_AGENT_SERVERS_FILE}" 2>/dev/null | awk '{print $1}')"
    python3 "${HM_EXAMPLE_DIR}/launch/node_monitor.py" "${RUN_DIR}/node-${SLURM_JOB_ID}-$(hostname -s).csv" \
        "${_me:-http://127.0.0.1:${HM_AGENT_SERVER_PORT:-65500}}" "${HM_NODE_MONITOR_S:-30}" >/dev/null 2>&1 &
fi
pid="$(cat "${RUN_DIR}/agent_servers/$(hostname -s).pid")"
hm_log "peer node: agent server pid ${pid} up; holding the step while it lives"
while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
hm_log "peer node: agent server ${pid} exited"
