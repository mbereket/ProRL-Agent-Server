#!/usr/bin/env bash
# Multi-node Ray cluster inside the Miles runtime, one call per Slurm node
# (srun --ntasks-per-node=1). The first node of the allocation is the head: it
# starts Ray, waits for every node, runs DRIVER on the head, then tears Ray
# down; the other nodes join and block until the head goes away.
#
#   ray_node.sh [mrun options, e.g. --patches DIR --bind X --pythonpath P] -- DRIVER [ARGS...]
#
# Inside DRIVER: RAY_ADDRESS=<head>:<gcs port>, MASTER_ADDR=<head ip>, RAY_DASHBOARD_URL,
# MILES_NUM_NODES. Everything (Ray daemons and DRIVER) runs inside the SIF with host
# networking, so Ray/NCCL/SGLang see the node exactly as bare metal would.
# Ports: RAY_GCS_PORT (6379), RAY_DASHBOARD_PORT (8265), RAY_CLIENT_PORT (10001).
set -euo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${MR}/lib.sh"

mrun_opts=()
while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; break ;;
        --no-nv) mrun_opts+=("$1"); shift ;;
        *) mrun_opts+=("$1" "$2"); shift 2 ;;
    esac
done
[ $# -gt 0 ] || mr_die "usage: ray_node.sh [mrun options] -- DRIVER [ARGS...]"

nodes=($(scontrol show hostnames "${SLURM_JOB_NODELIST:?not inside a slurm job}"))
NUM_NODES="${#nodes[@]}"
HEAD_HOST="${nodes[0]}"
HEAD_IP="$(getent ahostsv4 "${HEAD_HOST}" | awk 'NR==1{print $1}')"
[ -n "${HEAD_IP}" ] || mr_die "cannot resolve ${HEAD_HOST}"
GPUS="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"; RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-${USER}-${SLURM_JOB_ID}}"
# A busy raylet (CPU-offloaded optimizer, sandboxes) can lag heartbeats: tolerate ~5 min (hel 1026921).
export RAY_health_check_failure_threshold="${RAY_health_check_failure_threshold:-30}"
export RAY_health_check_timeout_ms="${RAY_health_check_timeout_ms:-30000}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
# Own PID namespace per node session: when the head's driver (or a worker's raylet) exits, every
# Ray/SGLang process of this session dies with it -- no stragglers holding GPUs for the next run.
export MRUN_APPTAINER_ARGS="${MRUN_APPTAINER_ARGS:---pid}"
export MASTER_ADDR="${HEAD_IP}" MILES_NUM_NODES="${NUM_NODES}"
export RAY_ADDRESS="${HEAD_IP}:${RAY_GCS_PORT}" RAY_DASHBOARD_URL="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"
mkdir -p "${RAY_TMPDIR}"
me="$(hostname -s)"

if [ "${me}" != "${HEAD_HOST%%.*}" ] && [ "${SLURM_NODEID:-0}" != 0 ]; then
    mr_log "worker ${me}: waiting for ray head ${HEAD_IP}:${RAY_GCS_PORT}"
    for _ in $(seq 1 600); do (echo > "/dev/tcp/${HEAD_IP}/${RAY_GCS_PORT}") 2>/dev/null && break; sleep 2; done
    exec "${MR}/mrun" "${mrun_opts[@]}" -- ray start --address="${RAY_ADDRESS}" --node-ip-address "$(getent ahostsv4 "${me}" | awk 'NR==1{print $1}')" \
        --num-gpus "${GPUS}" --disable-usage-stats --block
fi

mr_log "head ${me} (${HEAD_IP}): ${NUM_NODES} node(s) x ${GPUS} GPU"
# One container session holds the Ray head daemons and the driver.
driver="$(printf '%q ' "$@")"
exec "${MR}/mrun" "${mrun_opts[@]}" -- bash -c "
set -uo pipefail
ray start --head --node-ip-address '${HEAD_IP}' --port '${RAY_GCS_PORT}' --num-gpus '${GPUS}' \
    --dashboard-host 0.0.0.0 --dashboard-port '${RAY_DASHBOARD_PORT}' --disable-usage-stats >/dev/null
deadline=\$((SECONDS + \${RAY_JOIN_TIMEOUT:-900}))
until [ \"\$(python3 -c 'import ray; ray.init(address=\"auto\", logging_level=40); print(sum(1 for n in ray.nodes() if n[\"Alive\"]))' 2>/dev/null)\" = '${NUM_NODES}' ]; do
    [ \${SECONDS} -lt \${deadline} ] || { echo 'ray: nodes did not join in time' >&2; ray stop --force; exit 1; }
    sleep 5
done
echo \"[ray_node] cluster up: ${NUM_NODES} node(s)\"
${driver}
rc=\$?
ray stop --force >/dev/null 2>&1 || true
exit \${rc}
"
