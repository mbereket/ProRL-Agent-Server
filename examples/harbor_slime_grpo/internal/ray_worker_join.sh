#!/usr/bin/env bash
# Ray worker loop for non-head nodes: join the head's Ray cluster with this
# node's GPUs and stay joined (re-join if Ray on the head restarts).
#
#   bash internal/ray_worker_join.sh <head-ip> [gcs-port]
#
# Needs the same WORKROOT/ENV_FILE as the head (shared filesystem): it sources
# ENV_FILE so CUDA compat libs, the venv and HF_HOME match the head exactly.
# Under slurm this is started by head_entry.sh via srun; on a bare cluster run
# it by hand on every worker node before/while launch.sh runs on the head.
set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
HEAD_IP="${1:?usage: ray_worker_join.sh <head-ip> [gcs-port]}"
RAY_GCS_PORT="${2:-${RAY_GCS_PORT:-6379}}"
WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
ENV_FILE="${ENV_FILE:-${WORKROOT}/env.sh}"
NUM_GPUS="${RAY_WORKER_NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
export HF_HOME="${HF_HOME:-${WORKROOT}/hf_home}"

echo "[worker $(hostname)] waiting for ray head at ${HEAD_IP}:${RAY_GCS_PORT} (${NUM_GPUS} GPUs)"
while :; do
    if (echo > "/dev/tcp/${HEAD_IP}/${RAY_GCS_PORT}") 2>/dev/null; then
        sleep 5
        # The head has finished environment setup by the time Ray is up.
        # shellcheck disable=SC1090
        [ -f "${ENV_FILE}" ] && source "${ENV_FILE}"
        PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
        PATH="$(dirname "${PYTHON_BIN}"):${PATH}"; export PATH
        echo "[worker $(hostname)] joining ray at ${HEAD_IP}:${RAY_GCS_PORT}"
        ray start --address="${HEAD_IP}:${RAY_GCS_PORT}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --block || true
        echo "[worker $(hostname)] ray exited; re-polling"
        sleep 10
    else
        sleep 5
    fi
done
