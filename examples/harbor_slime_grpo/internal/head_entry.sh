#!/usr/bin/env bash
# Multi-node entry under slurm:  head_entry.sh <run-config.yaml>
# Runs once on the first node of the allocation: starts the Ray worker loop on
# every other node with srun, exports the multi-node addresses, then runs
# launch.sh <config> here. Needs a shared filesystem for the repo and WORKROOT.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${HERE}/.." && pwd)"
CONFIG="${1:?usage: head_entry.sh <run-config.yaml>}"
export PROJECT_ROOT="$(cd -- "${EXAMPLE_DIR}/../.." && pwd)"
export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"

resolve_ip() {   # `hostname -I` may list a link-local 169.254.* address first; prefer DNS.
    local ip; ip="$(getent hosts "$1" | awk '{print $1; exit}')"
    if [ -z "${ip}" ] || [[ "${ip}" == 169.254.* ]]; then
        ip="$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')"
    fi
    [ -n "${ip}" ] || { echo "ERROR: cannot resolve $1" >&2; exit 1; }
    echo "${ip}"
}
HEAD_IP="$(resolve_ip "$(hostname)")"
mapfile -t WORKERS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}" | grep -vx "$(hostname -s)" | grep -vx "$(hostname)")
WORKER_IP_LIST=(); for w in "${WORKERS[@]}"; do WORKER_IP_LIST+=("$(resolve_ip "${w}")"); done
export WORKER_HOSTS="$(IFS=,; echo "${WORKERS[*]}")" WORKER_IPS="$(IFS=,; echo "${WORKER_IP_LIST[*]}")"
export RAY_HEAD_IP="${HEAD_IP}" RAY_GCS_PORT="${RAY_GCS_PORT:-6379}" POLAR_BIND_HOST=0.0.0.0
export GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
echo "[head] $(hostname) (${HEAD_IP}); workers: ${WORKERS[*]:-none}"

# Workers source the run's env.sh (written by setup on the head) before joining Ray.
# shellcheck source=./setup/common.sh
source "${HERE}/setup/common.sh"
export ENV_FILE="$(config_python "${HERE}/render.py" env "${CONFIG}" | sed -n "s/^export RUN_DIR=//p" | tr -d "'")/env.sh"
WORKER_PIDS=()
for w in "${WORKERS[@]}"; do
    srun --overlap --nodes=1 --ntasks=1 -w "${w}" --gpus-per-node="${GPUS_PER_NODE}" \
        bash "${HERE}/ray_worker_join.sh" "${HEAD_IP}" "${RAY_GCS_PORT}" & WORKER_PIDS+=($!)
done
trap 'for p in "${WORKER_PIDS[@]}"; do kill "${p}" 2>/dev/null || true; done' EXIT

bash "${EXAMPLE_DIR}/launch.sh" "${CONFIG}"
