#!/usr/bin/env bash
# Start the processes of one run: Polar rollout server + one gateway per sandbox
# host, Ray, then the Slime trainer as a Ray job. Started by launch.sh after
# setup and render; needs its environment (RUN_DIR, ENV_FILE, PYTHON_BIN,
# SLIME_DIR, MEGATRON_DIR, NUM_NODES, GPUS_PER_NODE, SAVE_DIR, ...) and the
# rendered ${RUN_DIR}/{polar_config.yaml,topology.yaml,train_args.sh}.
#
# Services (ports are env knobs): POLAR_ROLLOUT_PORT (8080), POLAR_GATEWAY_PORT
# (8100), SGLANG_ROUTER_PORT (9000), RAY_DASHBOARD_PORT (8265), RAY_GCS_PORT (6379).
# Multi-node: head_entry.sh sets RAY_HEAD_IP, POLAR_BIND_HOST=0.0.0.0,
# WORKER_HOSTS and starts ray_worker_join.sh on every other node.
#
# Weight sync is GPU-to-GPU via NCCL every step. Slime manages the SGLang
# engines; the Polar gateway proxies agent LLM calls to them. With
# --dynamic-history every trace in an agent session becomes a training sample.
set -euo pipefail
: "${RUN_DIR:?}" "${ENV_FILE:?}" "${PYTHON_BIN:?}" "${SLIME_DIR:?}" "${MEGATRON_DIR:?}" "${SAVE_DIR:?}" "${PROJECT_ROOT:?}"
# shellcheck disable=SC1090
source "${ENV_FILE}"
# shellcheck disable=SC1091
source "${RUN_DIR}/train_args.sh"   # TRAIN_SCRIPT, TRAIN_ARGS, SANDBOX_IPS
cd "${PROJECT_ROOT}"

NUM_NODES="${NUM_NODES:-1}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-8080}"; POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-8100}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"; RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
TOPOLOGY="${RUN_DIR}/topology.yaml"
SESSION_ROOT="${RUN_DIR}/sessions"; mkdir -p "${SESSION_ROOT}" "${SAVE_DIR}"

# Compiler caches (Triton, Inductor, sglang JIT, flashinfer, tilelang) on node-local
# disk: many engines and trainer ranks compile concurrently and a cache on a network
# filesystem corrupts under that. Same path on every node; cold per job, optionally
# seeded from ${WORKROOT}/compiler_cache_seed on the head.
CACHE="/tmp/compiler_cache-${USER}-${SLURM_JOB_ID:-${RUN_ID}}"
SEED="${COMPILER_CACHE_SEED:-${WORKROOT}/compiler_cache_seed}"
[ ! -d "${CACHE}" ] && [ -d "${SEED}" ] && { mkdir -p "${CACHE}" && cp -r "${SEED}/." "${CACHE}/" && echo "compiler cache seeded from ${SEED}"; }
mkdir -p "${CACHE}"/{torchinductor,triton,tvm-ffi,sglang,tilelang,flashinfer} "${RUN_DIR}/wandb_cache"

# Ray actors on every node inherit exactly this environment: anything a worker
# node needs (CUDA compat libs, HF cache, toolkit) must be listed here. TMPDIR
# stays node-local: SGLang binds zmq IPC sockets there (Unix socket paths are
# capped at 107 chars, which a lustre run dir exceeds).
CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))')"
RUNTIME_ENV_JSON="$("${PYTHON_BIN}" -c 'import json, sys; print(json.dumps({"env_vars": dict(a.split("=", 1) for a in sys.argv[1:])}))' \
    "PYTHONPATH=${MEGATRON_DIR}:${PROJECT_ROOT}/src" \
    "PATH=$(dirname "${PYTHON_BIN}"):${PATH}" \
    "VIRTUAL_ENV=${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}" \
    "HF_HOME=${HF_HOME}" \
    "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}" \
    "LD_LIBRARY_PATH=${CUDNN_LIB}:${LD_LIBRARY_PATH:-}" \
    "CUDA_DEVICE_MAX_CONNECTIONS=1" \
    "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:True" \
    "TORCHINDUCTOR_CACHE_DIR=${CACHE}/torchinductor" \
    "TRITON_CACHE_DIR=${CACHE}/triton" \
    "TVM_FFI_CACHE_DIR=${CACHE}/tvm-ffi" \
    "SGLANG_CACHE_DIR=${CACHE}/sglang" \
    "TILELANG_CACHE_DIR=${CACHE}/tilelang" \
    "FLASHINFER_WORKSPACE_BASE=${CACHE}/flashinfer" \
    "FLASHINFER_USE_CUDA_NORM=${FLASHINFER_USE_CUDA_NORM:-1}" \
    "XDG_CACHE_HOME=${XDG_CACHE_HOME:-${WORKROOT}/xdg_cache}" \
    "TMPDIR=/tmp" \
    "WANDB_DIR=${RUN_DIR}/wandb_cache" "WANDB_CACHE_DIR=${RUN_DIR}/wandb_cache" "WANDB_DATA_DIR=${RUN_DIR}/wandb_cache" \
    "SLIME_ENGINE_BASE_PORT=${SLIME_ENGINE_BASE_PORT:-15000}" \
    "NVTE_DEBUG=1" "NVTE_DEBUG_LEVEL=2")"
# FLASHINFER_USE_CUDA_NORM=1: the default CuTe-DSL rmsnorm fails at SGLang cuda-graph
# capture on this stack (sglang 0.5.13 / flashinfer 0.6.12 / cutlass-dsl 4.5.2).

PIDS=()
cleanup() {
    echo "shutting down"
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ── Polar: rollout server on the head, one gateway per sandbox host ────────
echo "=== Polar rollout server :${POLAR_ROLLOUT_PORT}, gateway node-01 on $(hostname) :${POLAR_GATEWAY_PORT} ==="
polar serve_rollout -c "${TOPOLOGY}" & PIDS+=($!)
# Session dirs (agent logs, verifier output) under the run dir via POLAR_SESSION_DIR,
# not TMPDIR: apptainer forwards TMPDIR into the sandbox and breaks mktemp there.
export POLAR_KEEP_SESSION_DIRS="${POLAR_KEEP_SESSION_DIRS:-}"
POLAR_SESSION_DIR="${SESSION_ROOT}" polar serve_gateway -c "${TOPOLOGY}" --node-id node-01 & PIDS+=($!)
IFS=, read -r -a WORKER_HOST_LIST <<< "${WORKER_HOSTS:-}"
for ((i = 1; i < ${#SANDBOX_IPS[@]}; i++)); do
    host="${WORKER_HOST_LIST[$((i - 1))]}"; node_id="$(printf 'node-%02d' $((i + 1)))"
    echo "=== Polar gateway ${node_id} on ${host} ==="
    [ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: gateways on other hosts need a slurm allocation (srun)"; exit 1; }
    srun --overlap --nodes=1 --ntasks=1 -w "${host}" bash -c "source '${ENV_FILE}'; \
        export APPTAINER_CACHEDIR='${APPTAINER_CACHEDIR:-}' APPTAINER_TMPDIR='${APPTAINER_TMPDIR:-}' HF_HOME='${HF_HOME}' \
               POLAR_KEEP_SESSION_DIRS='${POLAR_KEEP_SESSION_DIRS}' POLAR_SESSION_DIR='${SESSION_ROOT}'; \
        cd '${PROJECT_ROOT}' && exec polar serve_gateway -c '${TOPOLOGY}' --node-id '${node_id}'" & PIDS+=($!)
done
sleep 3
curl -sf "http://127.0.0.1:${POLAR_ROLLOUT_PORT}/health" >/dev/null || { echo "ERROR: rollout server not healthy on :${POLAR_ROLLOUT_PORT}"; exit 1; }
for ((i = 0; i < ${#SANDBOX_IPS[@]}; i++)); do
    url="http://${SANDBOX_IPS[$i]}:${POLAR_GATEWAY_PORT}/health"; [ "$i" -eq 0 ] && url="http://127.0.0.1:${POLAR_GATEWAY_PORT}/health"
    for _ in $(seq 1 60); do curl -sf "${url}" >/dev/null 2>&1 && break; sleep 2; done
    curl -sf "${url}" >/dev/null || { echo "ERROR: gateway $(printf 'node-%02d' $((i + 1))) not healthy (${url})"; exit 1; }
    echo "gateway $(printf 'node-%02d' $((i + 1))) healthy"
done

# ── Ray ────────────────────────────────────────────────────────────────────
echo "=== Ray head on ${RAY_HEAD_IP} (${GPUS_PER_NODE} local GPUs, gcs :${RAY_GCS_PORT}) ==="
ray stop --force 2>/dev/null || true
ray start --head --node-ip-address "${RAY_HEAD_IP}" --port "${RAY_GCS_PORT}" --dashboard-port "${RAY_DASHBOARD_PORT}" \
    --num-gpus "${GPUS_PER_NODE}" --disable-usage-stats
if [ "${NUM_NODES}" -gt 1 ]; then
    deadline=$((SECONDS + ${RAY_JOIN_TIMEOUT:-900}))
    while :; do
        alive="$("${PYTHON_BIN}" -c 'import ray; ray.init(address="auto", logging_level="ERROR"); print(sum(n["Alive"] for n in ray.nodes()))' 2>/dev/null || echo 0)"
        [ "${alive}" -ge "${NUM_NODES}" ] && { echo "Ray cluster: ${alive} nodes alive"; break; }
        [ "${SECONDS}" -lt "${deadline}" ] || { echo "ERROR: only ${alive}/${NUM_NODES} Ray nodes joined"; exit 1; }
        sleep 10
    done
fi

# ── Checkpoint pruning (training.checkpoint_keep_every) ────────────────────
# Delete saved iterations that are neither a multiple of N nor the latest, every
# 5 min, so a small save_interval (periodic eval) does not accumulate ~180 GB/step.
if [ "${CHECKPOINT_KEEP_EVERY:-0}" -gt 0 ]; then
    ( while :; do
        latest="$(tr -dc '0-9' < "${SAVE_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || true)"
        for d in "${SAVE_DIR}"/iter_*/; do
            [ -d "${d}" ] && [ -n "${latest}" ] || continue
            it="$(basename "${d}" | sed 's/iter_0*//')"; it="${it:-0}"
            if [ "${it}" -lt "${latest}" ] && [ $((it % CHECKPOINT_KEEP_EVERY)) -ne 0 ]; then
                echo "[prune] removing $(basename "${d}") (latest ${latest}, keep every ${CHECKPOINT_KEEP_EVERY})"; rm -rf "${d}"
            fi
        done
        sleep 300
      done ) & PIDS+=($!)
fi

# ── Trainer ────────────────────────────────────────────────────────────────
echo "=== ${TRAIN_SCRIPT}: ${#TRAIN_ARGS[@]} args (${RUN_DIR}/train_args.sh); save ${SAVE_DIR} ==="
# The Ray dashboard binds loopback, so submission happens on the head. Unbuffered
# so the driver log (bridge drop reasons, slime step metrics) streams into the job log.
PYTHONUNBUFFERED=1 ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
