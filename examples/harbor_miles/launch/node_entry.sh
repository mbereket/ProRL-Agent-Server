#!/usr/bin/env bash
# One call per Slurm node (srun --ntasks-per-node=1): start this node's Harbor agent
# server on the host, then hand the node to the Miles runtime's Ray launcher
# (miles_runtime/ray_node.sh, STACK): the head runs launch/train_driver.sh inside the
# Miles SIF, the other nodes join Ray. Sandboxes therefore run on every node, next to
# (not inside) the containerized trainer.
#
#   node_entry.sh <config.env>      (config: see configs/*.env; sourced on every node)
#
# Needs MILES_RUNTIME_DIR = the miles_runtime package dir (uploaded next to this one).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
CFG="${1:?config.env}"
[ -f "${CFG}" ] || CFG="${HM_EXAMPLE_DIR}/${CFG}"
set -a; source "${CFG}"; set +a
: "${RUN_NAME:?config must set RUN_NAME}" "${HARBOR_TASKS_DIR:?config must set HARBOR_TASKS_DIR}"
export RUN_DIR="${HM_ROOT}/runs/${RUN_NAME}"
# Chained jobs (same RUN_NAME) share RUN_DIR; per-job files are keyed by SLURM_JOB_ID.
export HARBOR_AGENT_SERVERS_FILE="${RUN_DIR}/agent_servers-${SLURM_JOB_ID}.txt"
mkdir -p "${RUN_DIR}"
MILES_RUNTIME_DIR="${MILES_RUNTIME_DIR:-${SCOMPOSE_PKGS:-}/miles_runtime}"
[ -x "${MILES_RUNTIME_DIR}/ray_node.sh" ] || hm_die "miles_runtime not found at ${MILES_RUNTIME_DIR}"

# Idle-GPU reaper exemption (hel): trainer GPUs idle while long agent rollouts run.
if [ -n "${HM_REAPER_EXEMPT_MINS:-}" ] && [ "${SLURM_NODEID:-0}" = 0 ] && command -v scontrol >/dev/null; then
    scontrol update JobId="${SLURM_JOB_ID}" Comment="{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"${HM_REAPER_EXEMPT_MINS}\",\"reason\":\"other\",\"description\":\"asynchronous agentic RL: trainer GPUs idle while sandboxed agent rollouts run\"}}" \
        && hm_log "reaper exemption ${HM_REAPER_EXEMPT_MINS} min" || hm_log "WARNING: could not set reaper exemption"
fi

# Host side: this node's sandboxes.
HM_AGENT_TIMEOUT="${HM_AGENT_TIMEOUT:-3600}" \
    bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${RUN_DIR}" "${HM_AGENT_SERVER_PORT:-18300}" \
    "${HM_SANDBOXES_PER_NODE:-32}"

# Everything the rollout process (agent function) needs is in the environment Ray inherits.
export AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-qwen35-9b}"
export AGENT_TRIAL_TIMEOUT="${AGENT_TRIAL_TIMEOUT:-$(( ${HM_AGENT_TIMEOUT:-3600} + 1800 ))}"
# Container-side temp on node-local disk (common.sh points TMPDIR at lustre for host tools).
export TMPDIR="/tmp/hm-${USER}-${SLURM_JOB_ID}"; mkdir -p "${TMPDIR}"
exec bash "${MILES_RUNTIME_DIR}/ray_node.sh" \
    --pythonpath "${HM_EXAMPLE_DIR}/miles_side" --pythonpath "${HM_EXAMPLE_DIR}" \
    ${MILES_PATCH_DIR:+--patches "${MILES_PATCH_DIR}"} \
    -- bash "${HM_EXAMPLE_DIR}/launch/train_driver.sh"
