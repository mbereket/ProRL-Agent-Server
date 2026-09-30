#!/usr/bin/env bash
# Slurm entry:  head_entry.sh <run-config.yaml>
# Runs once on the first node of the allocation: exports the multi-node addresses
# (head IP, worker hosts/IPs for the Polar gateways), then runs launch.sh here.
# Ray itself is started on every node by run.sh (srun + miles_runtime/ray_node.sh),
# so there is no separate worker loop. Needs a shared filesystem for the repo and WORKROOT.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
EXAMPLE_DIR="$(cd -- "${HERE}/.." && pwd)"
CONFIG="${1:?usage: head_entry.sh <run-config.yaml>}"

resolve_ip() {   # `hostname -I` may list a link-local 169.254.* address first; prefer DNS.
    local ip; ip="$(getent ahostsv4 "$1" | awk 'NR==1{print $1}')"
    if [ -z "${ip}" ] || [[ "${ip}" == 169.254.* ]]; then
        ip="$(ip route get 8.8.8.8 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')"
    fi
    [ -n "${ip}" ] || { echo "ERROR: cannot resolve $1" >&2; exit 1; }
    echo "${ip}"
}
mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?head_entry.sh runs inside a slurm allocation}")
# ray_node.sh makes the first node of the allocation the Ray head; this script must run there.
[ "$(hostname -s)" = "${NODES[0]%%.*}" ] || { echo "ERROR: head_entry.sh must run on ${NODES[0]}, not $(hostname -s)" >&2; exit 1; }
HEAD_IP="$(resolve_ip "${NODES[0]}")"
WORKERS=("${NODES[@]:1}")
WORKER_IP_LIST=(); for w in "${WORKERS[@]}"; do WORKER_IP_LIST+=("$(resolve_ip "${w}")"); done
export WORKER_HOSTS="$(IFS=,; echo "${WORKERS[*]}")" WORKER_IPS="$(IFS=,; echo "${WORKER_IP_LIST[*]}")"
export RAY_HEAD_IP="${HEAD_IP}" POLAR_BIND_HOST=0.0.0.0
export GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
echo "[head] $(hostname) (${HEAD_IP}); workers: ${WORKERS[*]:-none}"
exec bash "${EXAMPLE_DIR}/launch.sh" "${CONFIG}"
