#!/usr/bin/env bash
# Multi-node entry point under slurm:  head_entry.sh <run-config.yaml>
# Runs once, on the first node of the allocation (the sbatch batch shell):
# starts the Ray worker loop on every other node with srun, exports the
# multi-node network settings, then runs launch.sh <config> here.
#
# Requires: a shared filesystem for the repo and WORKROOT; SLURM_JOB_NODELIST.
# The GPU layout (cluster.num_nodes, actor_num_gpus, tp_size,
# context_parallel_size) comes from the run config; on more than one node the
# trainer must take whole nodes (slime v0.3.0), every other GPU serves an
# SGLang engine (2 nodes → 8 train / 8 serve, 3 nodes → 8 train / 16 serve).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_CONFIG="${1:?usage: head_entry.sh <run-config.yaml>}"

routable_ip() {
    # `hostname -I` may list a link-local 169.254.* address first; prefer DNS.
    local ip; ip="$(getent hosts "$(hostname)" | awk '{print $1; exit}')"
    if [ -z "${ip}" ] || [[ "${ip}" == 169.254.* ]]; then
        ip="$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')"
    fi
    echo "${ip}"
}

HEAD_IP="$(routable_ip)"
[ -n "${HEAD_IP}" ] || { echo "ERROR: could not determine a routable head IP" >&2; exit 1; }
mapfile -t WORKERS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -vx "$(hostname -s)" | grep -vx "$(hostname)")
# launch.sh checks cluster.num_nodes in the config against this allocation.
NUM_NODES="${SLURM_JOB_NUM_NODES}"
[ "${#WORKERS[@]}" -ge $((NUM_NODES - 1)) ] || { echo "ERROR: need ${NUM_NODES} nodes, allocation has $((${#WORKERS[@]} + 1))" >&2; exit 1; }
echo "[head] $(hostname) (${HEAD_IP}); workers: ${WORKERS[*]:-none}"
# Worker hostnames and routable IPs, for run.sh to place Polar gateway nodes
# (cluster.sandbox_nodes: all) on them.
WORKER_IPS_LIST=()
for w in "${WORKERS[@]}"; do
    ip="$(getent hosts "${w}" | awk '{print $1; exit}')"
    [ -n "${ip}" ] || { echo "ERROR: cannot resolve worker ${w}" >&2; exit 1; }
    WORKER_IPS_LIST+=("${ip}")
done
export WORKER_HOSTS="$(IFS=,; echo "${WORKERS[*]}")"
export WORKER_IPS="$(IFS=,; echo "${WORKER_IPS_LIST[*]}")"

export RAY_HEAD_IP="${HEAD_IP}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export POLAR_BIND_HOST=0.0.0.0
export POLAR_PUBLIC_HOST="${HEAD_IP}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
export WORKROOT="${WORKROOT:-$(cd -- "${EXAMPLE_DIR}/../.." && pwd)/tmp}"
# The persisted setup environment is per run (pipeline.sh writes
# ${RUN_DIR}/env.sh); workers source the same file, so derive the path here.
PROJECT_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"; export PROJECT_ROOT   # config_python (common.sh) needs it
# shellcheck source=./setup/common.sh
source "${SCRIPT_DIR}/setup/common.sh"
RUN_NAME_CFG="$(config_python "${SCRIPT_DIR}/config_to_env.py" "${RUN_CONFIG}" | sed -n 's/^export RUN_NAME=//p' | tr -d "'")"
export ENV_FILE="${ENV_FILE:-${WORKROOT}/harbor_slime_grpo/${RUN_ID:-${RUN_NAME_CFG}}/env.sh}"

WORKER_PIDS=()
for w in "${WORKERS[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${w}" --gpus-per-node="${SLURM_GPUS_PER_NODE:-8}" \
        bash "${SCRIPT_DIR}/ray_worker_join.sh" "${HEAD_IP}" "${RAY_GCS_PORT}" &
    WORKER_PIDS+=($!)
done
cleanup() { for pid in "${WORKER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT

bash "${EXAMPLE_DIR}/launch.sh" "${RUN_CONFIG}"
