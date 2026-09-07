#!/usr/bin/env bash
# Ray worker loop for non-head nodes: join the head's Ray cluster with this
# node's GPUs and stay joined (re-join if Ray on the head restarts).
#
#   bash ray_worker_join.sh <head-ip> [gcs-port]      (started by head_entry.sh via srun)
#
# Sources ENV_FILE (the run's env.sh, shared filesystem) so CUDA compat libs, the
# venv and HF_HOME match the head exactly.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${HERE}/../../.." && pwd)}"
HEAD_IP="${1:?usage: ray_worker_join.sh <head-ip> [gcs-port]}"
RAY_GCS_PORT="${2:-${RAY_GCS_PORT:-6379}}"
: "${ENV_FILE:?}"
NUM_GPUS="${RAY_WORKER_NUM_GPUS:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
export HF_HOME="${HF_HOME:-${WORKROOT:?}/hf_home}"
# Same heartbeat tolerance as the head (run.sh): a raylet on a busy trainer node must not be declared dead after 60 s.
export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-30}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-30000}"

echo "[worker $(hostname)] waiting for ray head at ${HEAD_IP}:${RAY_GCS_PORT} (${NUM_GPUS} GPUs)"
while :; do
    if (echo > "/dev/tcp/${HEAD_IP}/${RAY_GCS_PORT}") 2>/dev/null; then
        sleep 5
        # The head has finished setup by the time Ray is up.
        # shellcheck disable=SC1090
        source "${ENV_FILE}"
        PATH="$(dirname "${PYTHON_BIN}"):${PATH}"; export PATH
        echo "[worker $(hostname)] joining ray at ${HEAD_IP}:${RAY_GCS_PORT}"
        ray start --address="${HEAD_IP}:${RAY_GCS_PORT}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --block || true
        echo "[worker $(hostname)] ray exited; re-polling"
        sleep 10
    else
        sleep 5
    fi
done
