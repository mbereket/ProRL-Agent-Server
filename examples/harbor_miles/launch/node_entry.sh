#!/usr/bin/env bash
# One call per Slurm node (srun --ntasks-per-node=1): start this node's Harbor agent
# server on the host, then hand the node to the Miles runtime's Ray launcher
# (miles_runtime/ray_node.sh, STACK): the head runs launch/train_driver.sh inside the
# Miles SIF, the other nodes join Ray. Sandboxes therefore run on every node, next to
# (not inside) the containerized trainer.
#
#   node_entry.sh <experiment.env>  (configs/README.md: recipe > layout > dataset > experiment layers, hm_load_config)
#
# Needs MILES_RUNTIME_DIR = the miles_runtime package dir (uploaded next to this one).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
CFG="${1:?config.env}"
[ -f "${CFG}" ] || CFG="${HM_EXAMPLE_DIR}/${CFG}"
hm_load_config "${CFG}" || hm_die "config ${CFG}: see the message above"
: "${RUN_NAME:?config must set RUN_NAME}" "${HARBOR_TASKS_DIR:?config must set HARBOR_TASKS_DIR}"
export RUN_DIR="${HM_RUNS_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
hm_log "$(hm_config_summary)"
# Run-dir lock: a run never executes twice at once (chained resubmits, dual-cluster twins sharing a
# filesystem). Held by the head node for the job's lifetime (fd survives the exec below).
if [ "${SLURM_NODEID:-0}" = 0 ]; then
    exec 9>"${RUN_DIR}/run.lock"
    flock -n 9 || hm_die "run ${RUN_NAME} is already running (lock ${RUN_DIR}/run.lock held)"
    echo "${SLURM_JOB_ID} $(hostname -s) $(date +%s)" > "${RUN_DIR}/run.owner"
    # jobs.log: one line per job of this run (cluster, nodes, Slurm log, config layers) -> where to look for what.
    _joblog="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/^ *StdOut=//p' | head -1)"
    echo "$(date '+%F %T') job=${SLURM_JOB_ID} cluster=${CLUSTER} nodes=${SLURM_NNODES:-1} chain=${HM_CHAIN_INDEX:-1}/${HM_CHAIN_MAX:-0} log=${_joblog:--} config=${HM_CONFIG_LAYERS:-${CFG}}" >> "${RUN_DIR}/jobs.log"
    hm_config_summary > "${RUN_DIR}/config-${SLURM_JOB_ID}.txt"
    # Cluster-side chaining (HM_CHAIN_MAX): queue the successor chunk now (launch/chain.sh).
    source "${HM_EXAMPLE_DIR}/launch/chain.sh"
    hm_chain
fi
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

# Other nodes' sandboxes (multi-node). The slurm-compose step runs this script on node 0 only, and ray_node.sh starts
# only Ray workers elsewhere, so the head starts every other node's agent server + node monitor itself: one overlapping
# 1-task srun step per node (launch/node_peer.sh; it blocks while its agent server lives; the job's end tears it down).
# Skipped when this script already runs once per node (srun --ntasks-per-node=1: each node starts its own).
if [ "${SLURM_NNODES:-1}" -gt 1 ] && [ "${SLURM_STEP_NUM_TASKS:-1}" -le 1 ] && command -v srun >/dev/null; then
    mkdir -p "${RUN_DIR}/agent_servers"
    for _n in $(scontrol show hostnames "${SLURM_JOB_NODELIST}" | tail -n +2); do
        srun --overlap --nodes=1 --ntasks=1 -w "${_n}" --cpus-per-task="${SLURM_CPUS_PER_TASK:-96}" --kill-on-bad-exit=0 \
            bash "${HM_EXAMPLE_DIR}/launch/node_peer.sh" "$(readlink -f "${CFG}")" > "${RUN_DIR}/agent_servers/peer-${SLURM_JOB_ID}-${_n}.log" 2>&1 &
        hm_log "peer node ${_n}: agent server step started (log agent_servers/peer-${SLURM_JOB_ID}-${_n}.log)"
    done
fi

# Host side: this node's sandboxes.
HM_AGENT_TIMEOUT="${HM_AGENT_TIMEOUT:-3600}" \
    bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${RUN_DIR}" "${HM_AGENT_SERVER_PORT:-65500}" \
    "${HM_SANDBOXES_PER_NODE:-32}"

# Per-node CPU / memory / sandbox sampler (sandbox-density measurements): RUN_DIR/node-<job>-<host>.csv.
if [ "${HM_NODE_MONITOR:-1}" = 1 ] && command -v python3 >/dev/null; then
    _me="$(grep -m1 "//$(hm_node_ip):" "${HARBOR_AGENT_SERVERS_FILE}" 2>/dev/null | awk '{print $1}')"
    python3 "${HM_EXAMPLE_DIR}/launch/node_monitor.py" "${RUN_DIR}/node-${SLURM_JOB_ID}-$(hostname -s).csv" \
        "${_me:-http://127.0.0.1:${HM_AGENT_SERVER_PORT:-65500}}" "${HM_NODE_MONITOR_S:-30}" >/dev/null 2>&1 &
fi

# Everything the rollout process (agent function) needs is in the environment Ray inherits.
export AGENT_MODEL_NAME="${AGENT_MODEL_NAME:-qwen35-9b}"
export HM_TRIAL_LOG="${HM_TRIAL_LOG:-${RUN_DIR}/trials-${SLURM_JOB_ID}.jsonl}"
# Miles patches (setup-time, pinned upstream + patch files): session /v1/responses for codex.
export MILES_PATCH_DIR="${MILES_PATCH_DIR:-${HM_EXAMPLE_DIR}/miles_patches}"
# Per-turn output cap for Responses clients that send none (codex). Default 0 = none: the engine then stops a turn only
# at the context window, which the session adapter reports to the agent as context_length_exceeded (overlong).
export MILES_RESPONSES_DEFAULT_MAX_TOKENS="${MILES_RESPONSES_DEFAULT_MAX_TOKENS:-0}"
export AGENT_TRIAL_TIMEOUT="${AGENT_TRIAL_TIMEOUT:-$(( ${HM_AGENT_TIMEOUT:-3600} + 1800 ))}"
# LORA_SERVE=merged (recipe default): STACK's train-LoRA/serve-merged patch set goes before ours.
[ "${LORA_SERVE:-merged}" = merged ] && MERGED_PATCHES="${MILES_RUNTIME_DIR}/patches/lora-serve-merged" || MERGED_PATCHES=""
# NO_MTP (DEFAULT 1): the bridge-built MTP layer is dropped unless trained. That removes an unweighted 0.2 next-token loss
# from the LoRA gradient and the fp32 MTP logits that OOM 27B at 128k (FINDINGS "MTP loss"). Since miles-stack a49a88ab the
# drop is base patch 0004 (always on) and patches/no-mtp is an empty alias kept for older runtimes; with the current runtime
# MTP comes back only with EXTRA=--enable-mtp-training. Runs before 2026-10-01 ~06:00 had the MTP loss ON.
[ "${NO_MTP:-1}" = 1 ] && MERGED_PATCHES="${MERGED_PATCHES:+${MERGED_PATCHES}:}${MILES_RUNTIME_DIR}/patches/no-mtp"
hm_log "patch sets: ${MERGED_PATCHES:-none} + ${MILES_PATCH_DIR:-none} (NO_MTP=${NO_MTP:-1})"
exec bash "${MILES_RUNTIME_DIR}/ray_node.sh" \
    --pythonpath "${HM_EXAMPLE_DIR}/miles_side" --pythonpath "${HM_EXAMPLE_DIR}" \
    $(IFS=:; for d in ${MERGED_PATCHES}; do printf -- '--patches %s ' "$d"; done) ${MILES_PATCH_DIR:+--patches "${MILES_PATCH_DIR}"} \
    -- bash "${HM_EXAMPLE_DIR}/launch/train_driver.sh"
