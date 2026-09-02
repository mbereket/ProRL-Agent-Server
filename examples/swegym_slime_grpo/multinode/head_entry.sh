#!/usr/bin/env bash
# Multi-node entry point under slurm. Runs once, on the first node of the
# allocation (the sbatch batch shell): starts the Ray worker loop on every other
# node with srun, exports the multi-node knobs, then runs launch_e2e.sh here.
#
# Requires: a shared filesystem for the repo and WORKROOT; SLURM_JOB_NODELIST.
# Knobs (defaults): ACTOR_NUM_NODES (=allocation size), TP_SIZE (2),
# CONTEXT_PARALLEL_SIZE (all train GPUs / TP, i.e. DP=1 for maximum context),
# ACTOR_NUM_GPUS_PER_NODE (4), ROLLOUT_NUM_GPUS_PER_NODE (4).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

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
export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${SLURM_JOB_NUM_NODES}}"
[ "${#WORKERS[@]}" -ge $((ACTOR_NUM_NODES - 1)) ] || { echo "ERROR: need ${ACTOR_NUM_NODES} nodes, allocation has $((${#WORKERS[@]} + 1))" >&2; exit 1; }
echo "[head] $(hostname) (${HEAD_IP}); workers: ${WORKERS[*]:-none}"

export RAY_HEAD_IP="${HEAD_IP}"
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export POLAR_BIND_HOST=0.0.0.0
export POLAR_PUBLIC_HOST="${HEAD_IP}"
export ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
export ROLLOUT_NUM_GPUS_PER_NODE="${ROLLOUT_NUM_GPUS_PER_NODE:-4}"
export TP_SIZE="${TP_SIZE:-2}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-$((ACTOR_NUM_GPUS_PER_NODE * ACTOR_NUM_NODES / TP_SIZE))}"
export WORKROOT="${WORKROOT:-$(cd -- "${EXAMPLE_DIR}/../.." && pwd)/tmp}"
export ENV_FILE="${ENV_FILE:-${WORKROOT}/env.sh}"

WORKER_PIDS=()
for w in "${WORKERS[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${w}" --gpus-per-node="${SLURM_GPUS_PER_NODE:-8}" \
        bash "${SCRIPT_DIR}/ray_worker_join.sh" "${HEAD_IP}" "${RAY_GCS_PORT}" &
    WORKER_PIDS+=($!)
done
cleanup() { for pid in "${WORKER_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done; }
trap cleanup EXIT

bash "${EXAMPLE_DIR}/launch_e2e.sh"
