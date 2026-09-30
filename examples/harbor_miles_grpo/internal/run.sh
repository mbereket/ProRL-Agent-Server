#!/usr/bin/env bash
# Start the processes of one run: Polar rollout server + one gateway per sandbox
# host (host venv), then Ray + the Miles trainer inside the Miles image on every
# node of the slurm allocation (examples/miles_runtime/ray_node.sh). Started by
# launch.sh after setup and render; needs its environment (RUN_DIR, ENV_FILE,
# PROJECT_ROOT, POLAR_PYTHON, MILES_RUNTIME_DIR, NUM_NODES, GPUS_PER_NODE, SAVE_DIR)
# and the rendered ${RUN_DIR}/{polar_config.yaml,topology.yaml,train_args.sh}.
#
# Ports (env knobs): POLAR_ROLLOUT_PORT (8080), POLAR_GATEWAY_PORT (8100),
# SGLANG_ROUTER_PORT (9000, the Miles/SGLang model gateway the Polar gateways call),
# RAY_GCS_PORT (6379), RAY_DASHBOARD_PORT (8265).
set -euo pipefail
: "${RUN_DIR:?}" "${ENV_FILE:?}" "${PROJECT_ROOT:?}" "${SAVE_DIR:?}"
# shellcheck disable=SC1090
source "${ENV_FILE}"
: "${POLAR_PYTHON:?}" "${MILES_RUNTIME_DIR:?}"
# shellcheck disable=SC1091
source "${RUN_DIR}/train_args.sh"   # TRAIN_SCRIPT, TRAIN_ARGS, SANDBOX_IPS
cd "${PROJECT_ROOT}"
export HF_HUB_OFFLINE=1
[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: run.sh needs a slurm allocation (Ray runs on every node via srun)"; exit 1; }

NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-8080}"; POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-8100}"
TOPOLOGY="${RUN_DIR}/topology.yaml"
SESSION_ROOT="${RUN_DIR}/sessions"; mkdir -p "${SESSION_ROOT}" "${SAVE_DIR}"
POLAR=("${POLAR_PYTHON}" -m polar.cli)
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PIDS=()
cleanup() {
    echo "shutting down"
    for pid in "${PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ── Polar: rollout server on the head, one gateway per sandbox host ────────
echo "=== Polar rollout server :${POLAR_ROLLOUT_PORT}, gateway node-01 on $(hostname) :${POLAR_GATEWAY_PORT} ==="
"${POLAR[@]}" serve_rollout -c "${TOPOLOGY}" & PIDS+=($!)
for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:${POLAR_ROLLOUT_PORT}/health" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://127.0.0.1:${POLAR_ROLLOUT_PORT}/health" >/dev/null || { echo "ERROR: rollout server not healthy on :${POLAR_ROLLOUT_PORT}"; exit 1; }
# Session dirs under the run dir via POLAR_SESSION_DIR, not TMPDIR (apptainer forwards
# TMPDIR into the sandbox and breaks mktemp there).
export POLAR_KEEP_SESSION_DIRS="${POLAR_KEEP_SESSION_DIRS:-}"
POLAR_SESSION_DIR="${SESSION_ROOT}" "${POLAR[@]}" serve_gateway -c "${TOPOLOGY}" --node-id node-01 & PIDS+=($!)
IFS=, read -r -a WORKER_HOST_LIST <<< "${WORKER_HOSTS:-}"
for ((i = 1; i < ${#SANDBOX_IPS[@]}; i++)); do
    host="${WORKER_HOST_LIST[$((i - 1))]}"; node_id="$(printf 'node-%02d' $((i + 1)))"
    echo "=== Polar gateway ${node_id} on ${host} ==="
    srun --overlap --nodes=1 --ntasks=1 -w "${host}" bash -c "source '${ENV_FILE}'; \
        export APPTAINER_CACHEDIR='${APPTAINER_CACHEDIR:-}' APPTAINER_TMPDIR='${APPTAINER_TMPDIR:-}' HF_HOME='${HF_HOME}' HF_HUB_OFFLINE=1 \
               POLAR_KEEP_SESSION_DIRS='${POLAR_KEEP_SESSION_DIRS}' POLAR_SESSION_DIR='${SESSION_ROOT}' PYTHONPATH='${PYTHONPATH}'; \
        cd '${PROJECT_ROOT}' && exec '${POLAR_PYTHON}' -m polar.cli serve_gateway -c '${TOPOLOGY}' --node-id '${node_id}'" & PIDS+=($!)
done
for ((i = 0; i < ${#SANDBOX_IPS[@]}; i++)); do
    url="http://${SANDBOX_IPS[$i]}:${POLAR_GATEWAY_PORT}/health"; [ "$i" -eq 0 ] && url="http://127.0.0.1:${POLAR_GATEWAY_PORT}/health"
    for _ in $(seq 1 60); do curl -sf "${url}" >/dev/null 2>&1 && break; sleep 2; done
    curl -sf "${url}" >/dev/null || { echo "ERROR: gateway $(printf 'node-%02d' $((i + 1))) not healthy (${url})"; exit 1; }
    echo "gateway $(printf 'node-%02d' $((i + 1))) healthy: $(curl -sf "${url%/health}/admin/inference/status" || true)"
done

# ── Ray + Miles trainer, inside the Miles image on every node ──────────────
# ray_node.sh: node 0 (this host) starts the Ray head and runs the driver in the
# same container session; the others join. Bridge + Polar code come from
# ${PROJECT_ROOT}/src (--pythonpath); patch sets under patches/ are layered on the
# image's Miles / SGLang / Megatron trees by mrun.
mrun_opts=(--pythonpath "${PROJECT_ROOT}/src")
PATCH_ROOT="${PROJECT_ROOT}/examples/harbor_miles_grpo/patches"
if compgen -G "${PATCH_ROOT}/*/*.patch" >/dev/null || [ -f "${PATCH_ROOT}/pip-requirements.txt" ]; then
    mrun_opts+=(--patches "${PATCH_ROOT}")
fi
[ -n "${MILES_EXTRA_PATCHES:-}" ] && mrun_opts+=(--patches "${MILES_EXTRA_PATCHES}")
# Ray actors inherit the raylet environment (apptainer passes the host env through).
# TMPDIR stays node-local: SGLang binds zmq IPC sockets there (paths capped at 107 chars).
export TMPDIR=/tmp POLAR_TRAINER_FRAMEWORK=miles
export WANDB_DIR="${RUN_DIR}/wandb_cache" WANDB_CACHE_DIR="${RUN_DIR}/wandb_cache" WANDB_DATA_DIR="${RUN_DIR}/wandb_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${RUN_DIR}/wandb_cache"
echo "=== ${TRAIN_SCRIPT}: ${#TRAIN_ARGS[@]} args (${RUN_DIR}/train_args.sh); ${NUM_NODES} node(s); save ${SAVE_DIR} ==="
srun --overlap --nodes="${NUM_NODES}" --ntasks-per-node=1 --gpus-per-node="${GPUS_PER_NODE}" --kill-on-bad-exit=1 \
    bash "${MILES_RUNTIME_DIR}/ray_node.sh" "${mrun_opts[@]}" -- \
    python3 "/root/miles/${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
