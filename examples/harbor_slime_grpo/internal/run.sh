#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# GRPO training on Harbor tasks via Polar + Slime. Started by launch.sh, which
# exports every knob below from the run config; run it directly only with the
# same environment (see launch.sh --dry-run for the exact variables).
#
# Qwen3.5 checkpoints are VLMs (Qwen3_5ForConditionalGeneration) with hybrid
# attention (1 full + 3 GatedDeltaNet linear per 4 layers); we train text-only.
#
# Services (all ports are env knobs; defaults in parentheses):
#   SGLANG_ROUTER_PORT  (9000)  slime-managed SGLang router
#   POLAR_ROLLOUT_PORT  (8080)  Polar rollout server
#   POLAR_GATEWAY_PORT  (8100)  Polar gateway node
#   RAY_DASHBOARD_PORT  (8265)  Ray dashboard / job submission (loopback)
#   RAY_GCS_PORT        (6379)  Ray GCS (workers join here)
#
# Multi-node: set NUM_NODES, RAY_HEAD_IP=<routable head IP>,
# POLAR_BIND_HOST=0.0.0.0, POLAR_PUBLIC_HOST=<head IP>, and start
# `internal/ray_worker_join.sh` on every other node (internal/head_entry.sh
# does all of this under slurm). This script always runs on the head.
#
# Weight sync: native GPU-to-GPU via NCCL every training step. Slime manages the
# SGLang engines; the Polar gateway proxies agent LLM calls to them. With
# --dynamic-history every trace in an agent session becomes a training sample.
# TRAIN_SCRIPT=train.py runs synchronously (rollout, then update, then weight
# sync, every step); train_async.py (default) overlaps the next rollout with the
# current update (one step off-policy, corrected by TIS).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
# Environment written by pipeline.sh (CUDA compat libs, toolkit, venv, apptainer).
ENV_FILE="${ENV_FILE:-${WORKROOT}/env.sh}"
# shellcheck disable=SC1090
[ -f "${ENV_FILE}" ] && source "${ENV_FILE}"
RUN_DIR="${RUN_DIR:-${WORKROOT}/harbor_slime_grpo/${RUN_ID:-run}}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

is_path_like() { case "$1" in /*|./*|../*|~*) return 0 ;; *) return 1 ;; esac; }

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); print(s.getsockname()[0]); s.close()
except Exception:
    try: print(socket.gethostbyname(socket.gethostname()))
    except Exception: print("127.0.0.1")
PY
}

# ── External deps ──────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_async.py}"
[ -f "${SLIME_DIR}/${TRAIN_SCRIPT}" ] || { echo "ERROR: ${TRAIN_SCRIPT} not found in ${SLIME_DIR} (run launch.sh, or set SLIME_DIR)"; exit 1; }
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
[ -d "${MEGATRON_DIR}/megatron" ] || { echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR} (run launch.sh, or set MEGATRON_DIR)"; exit 1; }

# ── Model ──────────────────────────────────────────────────────────
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-9B}"
REF_LOAD="${REF_LOAD:-${TORCH_DIST_DIR:-${WORKROOT}/checkpoints/${HF_CHECKPOINT##*/}_torch_dist}}"
RUN_ID="${RUN_ID:-harbor-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${WORKROOT}/ckpt/harbor_slime_grpo}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "$SAVE_DIR"
if is_path_like "$HF_CHECKPOINT" && [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"; exit 1
fi
if [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run bash examples/harbor_slime_grpo/internal/convert_weights.sh first."; exit 1
fi
# MODEL_ARGS_FILE: model_args_9b.sh (Qwen3.5-9B, default) or model_args.sh (4B); relative to this dir or absolute.
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-model_args_9b.sh}"
case "${MODEL_ARGS_FILE}" in /*) ;; *) MODEL_ARGS_FILE="${SCRIPT_DIR}/${MODEL_ARGS_FILE}" ;; esac
# shellcheck disable=SC1090
source "${MODEL_ARGS_FILE}"

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
if [ -n "${MODEL_LOAD_DIR:-}" ]; then
    # Explicit checkpoint (e.g. another run's save dir, for evaluation or forking).
    [ -f "$MODEL_LOAD_DIR/latest_checkpointed_iteration.txt" ] || { echo "ERROR: model.load_dir has no checkpoint: $MODEL_LOAD_DIR"; exit 1; }
    LOAD_DIR="$MODEL_LOAD_DIR"; START_ROLLOUT_ARGS=()
elif [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"; START_ROLLOUT_ARGS=()          # resume: slime derives start_rollout_id from the checkpoint
else
    # Fresh run from the converted reference checkpoint. Its "release" iteration
    # loads as 0, which slime would turn into start_rollout_id=1 and silently drop
    # one of the num_epoch x (prompts / batch) rollouts; start at 0 explicitly.
    LOAD_DIR="$REF_LOAD"; START_ROLLOUT_ARGS=(--start-rollout-id 0)
fi

# ── Data ───────────────────────────────────────────────────────────
# Prompt JSONL from prepare_tasks.py (launch.sh writes it into the run's asset dir).
PROMPT_DATA="${PROMPT_DATA:?PROMPT_DATA must point at the prompt JSONL from prepare_tasks.py}"
[ -f "$PROMPT_DATA" ] || { echo "ERROR: prompt data not found: $PROMPT_DATA"; exit 1; }

# ── Parallelism / sizing ───────────────────────────────────────────
# NUM_NODES Ray nodes with GPUS_PER_NODE GPUs each. The trainer takes
# ACTOR_NUM_GPUS of them; every remaining GPU serves an SGLang engine unless
# ROLLOUT_NUM_GPUS is set. Generation is usually the bottleneck, so give it the
# larger share.
#
# Slime v0.3.0 places engines assuming they fill whole nodes in rank order, so
# on more than one node the trainer must take whole nodes: ACTOR_NUM_GPUS must
# be a multiple of GPUS_PER_NODE (2 nodes → 8 train / 8 serve, 3 nodes →
# 8 train / 16 serve). On a single node any split works.
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')}"
ACTOR_NUM_GPUS="${ACTOR_NUM_GPUS:-4}"
if [ "${NUM_NODES}" -gt 1 ] && [ $((ACTOR_NUM_GPUS % GPUS_PER_NODE)) -ne 0 ]; then
    echo "ERROR: with NUM_NODES=${NUM_NODES}, ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS} must be a multiple of GPUS_PER_NODE=${GPUS_PER_NODE}:"
    echo "  slime assigns engine addresses per whole node, so a trainer sharing a node with engines"
    echo "  leaves engines on other nodes with the wrong host. Use ACTOR_NUM_GPUS=${GPUS_PER_NODE} (or a multiple)."
    exit 1
fi
if [ "${ACTOR_NUM_GPUS}" -ge "${GPUS_PER_NODE}" ]; then
    [ $((ACTOR_NUM_GPUS % GPUS_PER_NODE)) -eq 0 ] || { echo "ERROR: ACTOR_NUM_GPUS=${ACTOR_NUM_GPUS} must be a multiple of GPUS_PER_NODE=${GPUS_PER_NODE} when it spans nodes"; exit 1; }
    ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-$((ACTOR_NUM_GPUS / GPUS_PER_NODE))}"
    ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${GPUS_PER_NODE}}"
else
    ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
    ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${ACTOR_NUM_GPUS}}"
fi
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$((NUM_NODES * GPUS_PER_NODE - ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))}"
[ "${ROLLOUT_NUM_GPUS}" -ge 1 ] || { echo "ERROR: no GPUs left for rollout (${NUM_NODES}×${GPUS_PER_NODE} total, ${ACTOR_NUM_NODES}×${ACTOR_NUM_GPUS_PER_NODE} train)"; exit 1; }
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
TP_SIZE="${TP_SIZE:-4}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-2}"   # per-trace cap = MAX_TOKENS_PER_GPU × CP; TP×CP must divide the actor GPUs
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-16384}"     # 16384 is known to fit H100-80GB for 9B TP4; probe before raising
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"   # keep = MAX_TOKENS_PER_GPU × CP so no trace is censored
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-24000}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-8000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

# ── Services / addresses ───────────────────────────────────────────
POLAR_ROLLOUT_PORT="${POLAR_ROLLOUT_PORT:-8080}"
POLAR_GATEWAY_PORT="${POLAR_GATEWAY_PORT:-8100}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"
POLAR_BIND_HOST="${POLAR_BIND_HOST:-127.0.0.1}"
POLAR_PUBLIC_HOST="${POLAR_PUBLIC_HOST:-${RAY_HEAD_IP}}"
POLAR_CALLBACK_HOST="${POLAR_CALLBACK_HOST:-127.0.0.1}"
export POLAR_BIND_HOST POLAR_ROLLOUT_PORT POLAR_GATEWAY_PORT POLAR_CALLBACK_HOST
export POLAR_ROLLOUT_URL="http://${POLAR_PUBLIC_HOST}:${POLAR_ROLLOUT_PORT}"
export POLAR_GATEWAY_URL="http://${POLAR_PUBLIC_HOST}:${POLAR_GATEWAY_PORT}"
export MODEL_SERVED="${MODEL_SERVED:-${HF_CHECKPOINT}}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${WORKROOT}/harbor_sif_images}"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || command -v singularity || echo /usr/bin/apptainer)}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
COMPILER_CACHE_ROOT="${COMPILER_CACHE_ROOT:-${RUN_DIR}/compiler_cache}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILER_CACHE_ROOT}/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILER_CACHE_ROOT}/triton}"
# sglang JIT kernels (tvm-ffi) default to ~/.cache/tvm-ffi; keep them off $HOME too.
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-${COMPILER_CACHE_ROOT}/tvm-ffi}"
# tilelang (FLA GatedDeltaNet kernels in the trainer) defaults to ~/.tilelang/cache.
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${COMPILER_CACHE_ROOT}/tilelang}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TVM_FFI_CACHE_DIR" "$TILELANG_CACHE_DIR"
# wandb artifact staging on the run dir (a full $HOME breaks it). The Ray job's
# TMPDIR stays node-local /tmp: SGLang binds zmq IPC sockets there and Unix
# socket paths are capped at 107 chars, which a lustre run dir exceeds. Polar
# gateways must not inherit TMPDIR either (apptainer forwards it into the sandbox).
mkdir -p "${RUN_DIR}/wandb_cache"

# Render YAML templates: only the listed ${VARS} are expanded.
command -v envsubst >/dev/null || { echo "ERROR: envsubst not found (install gettext-base)"; exit 1; }
export RUN_DIR
TEMPLATE_VARS='${SGLANG_ROUTER_BASE_URL} ${APPTAINER_IMAGE_DIR} ${POLAR_BIND_HOST} ${POLAR_ROLLOUT_PORT} ${POLAR_GATEWAY_PORT} ${POLAR_ROLLOUT_URL} ${POLAR_GATEWAY_URL} ${POLAR_CALLBACK_HOST} ${MODEL_SERVED} ${RUN_DIR}'
mkdir -p "$(dirname "$TOPOLOGY_PATH")" "$(dirname "$CUSTOM_CONFIG_PATH")"
envsubst "$TEMPLATE_VARS" < "$TOPOLOGY_TEMPLATE"     > "$TOPOLOGY_PATH"
envsubst "$TEMPLATE_VARS" < "$POLAR_CONFIG_TEMPLATE" > "$CUSTOM_CONFIG_PATH"

# ── Sandbox hosts: where Polar gateway nodes (and so the agent sandboxes) run ──
# SANDBOX_NODES=head: the head only (default). SANDBOX_NODES=all: the head plus
# every worker in WORKER_HOSTS/WORKER_IPS (comma lists exported by head_entry.sh
# under slurm; set them by hand otherwise). max_run_workers is per host.
SANDBOX_NODES="${SANDBOX_NODES:-head}"
SANDBOX_HOSTS=("$(hostname)"); SANDBOX_IPS=("${POLAR_PUBLIC_HOST}")
if [ "${SANDBOX_NODES}" = all ] && [ -n "${WORKER_IPS:-}" ]; then
    IFS=, read -r -a _wh <<< "${WORKER_HOSTS:?WORKER_HOSTS must accompany WORKER_IPS}"
    IFS=, read -r -a _wi <<< "${WORKER_IPS}"
    [ "${#_wh[@]}" -eq "${#_wi[@]}" ] || { echo "ERROR: WORKER_HOSTS and WORKER_IPS differ in length"; exit 1; }
    SANDBOX_HOSTS+=("${_wh[@]}"); SANDBOX_IPS+=("${_wi[@]}")
elif [ "${SANDBOX_NODES}" != head ] && [ "${SANDBOX_NODES}" != all ]; then
    echo "ERROR: SANDBOX_NODES must be head or all (got ${SANDBOX_NODES})"; exit 1
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/expand_gateway_nodes.py" "${TOPOLOGY_PATH}" "${SANDBOX_IPS[@]}"

cat <<INFO
Using topology:       ${TOPOLOGY_PATH}
Using Polar config:   ${CUSTOM_CONFIG_PATH}
Polar rollout/gateway ${POLAR_ROLLOUT_URL} / ${POLAR_GATEWAY_URL} (bind ${POLAR_BIND_HOST})
SGLang router:        ${SGLANG_ROUTER_BASE_URL}
Apptainer:            ${POLAR_APPTAINER_BIN}; images ${APPTAINER_IMAGE_DIR}
Model / args:         ${HF_CHECKPOINT} / ${MODEL_ARGS_FILE}
Layout:               ${NUM_NODES} node(s) × ${GPUS_PER_NODE} GPUs: train ${ACTOR_NUM_NODES}×${ACTOR_NUM_GPUS_PER_NODE} (TP${TP_SIZE} CP${CONTEXT_PARALLEL_SIZE}), rollout ${ROLLOUT_NUM_GPUS} engine GPUs; ${MAX_TOKENS_PER_GPU} tok/GPU
Sandboxes:            ${#SANDBOX_HOSTS[@]} host(s) [${SANDBOX_NODES}]: ${SANDBOX_HOSTS[*]}
Trainer:              ${TRAIN_SCRIPT}; prompts ${PROMPT_DATA}
Run id / save dir:    ${RUN_ID} / ${SAVE_DIR}
INFO

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
cleanup() {
    [ -n "${PRUNE_PID:-}" ] && kill "${PRUNE_PID}" 2>/dev/null || true
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ── Step 1: Polar services (host, CPU only) ────────────────────────
echo "=== Starting Polar rollout server (:${POLAR_ROLLOUT_PORT}) ==="
polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2
# Gateway session dirs (agent logs, artifacts, overlay) go under the shared run
# dir instead of the node's /tmp so they can be read after the fact:
# ${RUN_DIR}/sessions/session-<id>/logs/agent/. POLAR_SESSION_DIR, not TMPDIR:
# apptainer forwards TMPDIR into the sandbox and breaks mktemp there.
SESSION_ROOT="${SESSION_ROOT:-${RUN_DIR}/sessions}"; mkdir -p "${SESSION_ROOT}"
echo "=== Starting Polar gateway node-01 on $(hostname) (:${POLAR_GATEWAY_PORT}) ==="
export POLAR_KEEP_SESSION_DIRS="${POLAR_KEEP_SESSION_DIRS:-}"   # 1 keeps ${SESSION_ROOT}/session-*/ (agent logs, verifier output)
POLAR_SESSION_DIR="${SESSION_ROOT}" polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id node-01 &
PIDS+=($!)
# Gateways on the other sandbox hosts: same venv/env (ENV_FILE) and repo, own
# node id. Under slurm via srun inside this allocation, else ssh.
remote_gateway() {   # remote_gateway HOST NODE_ID
    local host="$1" node_id="$2" cmd
    cmd="source '${ENV_FILE}' 2>/dev/null; export APPTAINER_CACHEDIR='${APPTAINER_CACHEDIR:-}' APPTAINER_TMPDIR='${APPTAINER_TMPDIR:-}' HF_HOME='${HF_HOME:-}' POLAR_KEEP_SESSION_DIRS='${POLAR_KEEP_SESSION_DIRS}'; cd '${PROJECT_ROOT}' && POLAR_SESSION_DIR='${SESSION_ROOT}' exec polar serve_gateway -c '${TOPOLOGY_PATH}' --node-id '${node_id}'"
    echo "=== Starting Polar gateway ${node_id} on ${host} ==="
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        srun --overlap --nodes=1 --ntasks=1 -w "${host}" bash -c "${cmd}" &
    else
        ssh -o BatchMode=yes "${host}" "${cmd}" &
    fi
    PIDS+=($!)
}
for ((i = 1; i < ${#SANDBOX_HOSTS[@]}; i++)); do
    remote_gateway "${SANDBOX_HOSTS[$i]}" "$(printf 'node-%02d' $((i + 1)))"
done
sleep 2
curl -sf "http://127.0.0.1:${POLAR_ROLLOUT_PORT}/health" >/dev/null || { echo "Polar rollout server not healthy on :${POLAR_ROLLOUT_PORT}"; exit 1; }
for ((i = 0; i < ${#SANDBOX_IPS[@]}; i++)); do
    url="http://${SANDBOX_IPS[$i]}:${POLAR_GATEWAY_PORT}/health"; [ "$i" -eq 0 ] && url="http://127.0.0.1:${POLAR_GATEWAY_PORT}/health"
    for _ in $(seq 1 60); do curl -sf "${url}" >/dev/null 2>&1 && break; sleep 2; done
    curl -sf "${url}" >/dev/null || { echo "Polar gateway on ${SANDBOX_HOSTS[$i]} not healthy (${url})"; exit 1; }
    echo "gateway $(printf 'node-%02d' $((i + 1))) healthy on ${SANDBOX_HOSTS[$i]}"
done

# ── Step 2: Ray + Slime (SGLang engines + training) ────────────────
# The head registers only its local GPUs; other nodes join via ray_worker_join.sh.
RAY_NUM_GPUS="${RAY_NUM_GPUS:-${GPUS_PER_NODE}}"
echo "=== Starting Ray head on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} local GPUs, gcs :${RAY_GCS_PORT}) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address "$RAY_HEAD_IP" --port "$RAY_GCS_PORT" --dashboard-port "$RAY_DASHBOARD_PORT" \
    --num-gpus "$RAY_NUM_GPUS" --disable-usage-stats

if [ "${NUM_NODES}" -gt 1 ]; then
    RAY_JOIN_TIMEOUT="${RAY_JOIN_TIMEOUT:-900}"
    echo "=== Waiting for ${NUM_NODES} Ray nodes (timeout ${RAY_JOIN_TIMEOUT}s) ==="
    deadline=$((SECONDS + RAY_JOIN_TIMEOUT))
    while :; do
        alive="$("${PYTHON_BIN}" -c 'import ray; ray.init(address="auto", logging_level="ERROR"); print(sum(1 for n in ray.nodes() if n["Alive"]))' 2>/dev/null || echo 0)"
        [ "${alive}" -ge "${NUM_NODES}" ] && { echo "Ray cluster: ${alive} nodes alive"; break; }
        [ "${SECONDS}" -ge "${deadline}" ] && { echo "ERROR: only ${alive}/${NUM_NODES} Ray nodes joined within ${RAY_JOIN_TIMEOUT}s"; exit 1; }
        sleep 10
    done
fi

# cuDNN lib path — probe the active venv.
if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
# Ray actors on every node inherit exactly this environment; anything a worker
# node needs (CUDA compat libs, HF cache, toolkit) must be listed here.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"HF_HOME\": \"${HF_HOME:-${HOME}/.cache/huggingface}\",
    \"CUDA_HOME\": \"${CUDA_HOME:-/usr/local/cuda}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TVM_FFI_CACHE_DIR\": \"${TVM_FFI_CACHE_DIR}\",
    \"TILELANG_CACHE_DIR\": \"${TILELANG_CACHE_DIR}\",
    \"TMPDIR\": \"/tmp\",
    \"WANDB_CACHE_DIR\": \"${RUN_DIR}/wandb_cache\",
    \"WANDB_DATA_DIR\": \"${RUN_DIR}/wandb_cache\",
    \"SLIME_ENGINE_BASE_PORT\": \"${SLIME_ENGINE_BASE_PORT:-15000}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"

# Rollout sizing: ROLLOUT_BATCH_SIZE prompts × N_SAMPLES_PER_PROMPT trajectories
# per rollout. With --dynamic-history each trajectory yields one sample per
# trace, so the sample count per rollout is variable. The custom data source
# rounds the epoch up so every train prompt is consumed once.
# Algorithm knobs (defaults = the original example): USE_KL_LOSS=0 drops the KL
# term, GRPO_STD_NORMALIZATION=1 re-enables std scaling (default: mean-only advantages, Dr.GRPO),
# EVAL_PROMPT_DATA="<name> <path>" adds a held-out eval every EVAL_INTERVAL steps,
# EXTRA_TRAIN_ARGS appends arbitrary train_async.py flags (whitespace-split).
KL_ARGS=(--use-kl-loss --kl-loss-coef "${KL_LOSS_COEF:-0.001}" --kl-loss-type low_var_kl)
[ "${USE_KL_LOSS:-1}" = 1 ] || KL_ARGS=()
STD_NORM_ARGS=()
[ "${GRPO_STD_NORMALIZATION:-0}" = 1 ] || STD_NORM_ARGS=(--disable-grpo-std-normalization)
# OPTIMIZER_CPU_OFFLOAD=1 keeps Adam states and fp32 master params on the host
# (Megatron hybrid optimizer). They are allocated at the first update, so a
# config that fits step 0 can OOM from step 1 on; with ~65k-token traces on
# TP4xCP2 that is ~20 GB/GPU of headroom.
OPTIM_OFFLOAD_ARGS=()
[ "${OPTIMIZER_CPU_OFFLOAD:-0}" = 1 ] && OPTIM_OFFLOAD_ARGS=(--optimizer-cpu-offload --optimizer-offload-fraction 1.0 --overlap-cpu-optimizer-d2h-h2d)
# NUM_ROLLOUT overrides the epoch-derived step count (slime then ignores --num-epoch);
# 0 runs only the eval pass (rollout.num_rollout: 0 + eval.prompt_data).
NUM_ROLLOUT_ARGS=()
if [ -n "${NUM_ROLLOUT:-}" ]; then
    NUM_ROLLOUT_ARGS=(--num-rollout "${NUM_ROLLOUT}")
    # slime sizes Megatron's LR schedule from num_rollout; with 0 the scheduler
    # asserts lr_decay_steps > 0. No optimizer step runs in eval-only mode.
    # Eval-only also skips the checkpoint's optimizer/RNG state: it is unused, and
    # a state saved from a different GPU layout (e.g. 8-GPU TP4xCP2) does not fit
    # when re-sharded onto a smaller eval allocation.
    [ "${NUM_ROLLOUT}" = 0 ] && NUM_ROLLOUT_ARGS+=(--lr-decay-iters 1 --no-load-optim --no-load-rng)
fi
# CHECKPOINT_KEEP_EVERY=N prunes saved iterations that are not multiples of N
# (the latest is always kept), every 5 min while training runs. Lets
# save_interval feed periodic evals without accumulating ~180 GB per step.
if [ "${CHECKPOINT_KEEP_EVERY:-0}" -gt 0 ]; then
    ( while :; do bash "${SCRIPT_DIR}/prune_checkpoints.sh" "${SAVE_DIR}" "${CHECKPOINT_KEEP_EVERY}" || true; sleep 300; done ) &
    PRUNE_PID=$!   # stopped by cleanup() on exit
fi
EVAL_ARGS=()
if [ -n "${EVAL_PROMPT_DATA:-}" ]; then
    # shellcheck disable=SC2206
    EVAL_ARGS=(--eval-prompt-data ${EVAL_PROMPT_DATA} --eval-interval "${EVAL_INTERVAL:-10}" --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}")
fi
# shellcheck disable=SC2206
EXTRA_TRAIN_ARGS_ARR=(${EXTRA_TRAIN_ARGS:-})

echo "=== Launching ${TRAIN_SCRIPT} ==="
# The Ray dashboard binds loopback; submission always happens on the head.
# Unbuffered so the driver log (bridge drop reasons, slime step metrics) streams
# into the job log instead of arriving in 8 KB chunks.
PYTHONUNBUFFERED=1 ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/${TRAIN_SCRIPT}" \
    --actor-num-nodes "$ACTOR_NUM_NODES" \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    ${START_ROLLOUT_ARGS[@]+"${START_ROLLOUT_ARGS[@]}"} \
    --save "$SAVE_DIR" \
    --save-interval "$SAVE_INTERVAL" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${CUSTOM_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-epoch "${NUM_EPOCH:-1}" \
    ${NUM_ROLLOUT_ARGS[@]+"${NUM_ROLLOUT_ARGS[@]}"} \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN" \
    --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN" \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size "$TP_SIZE" \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size "$CONTEXT_PARALLEL_SIZE" \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
    --log-probs-chunk-size 256 \
    --distributed-timeout-minutes 30 \
    --advantage-estimator grpo \
    --normalize-advantages \
    --use-tis \
    "${KL_ARGS[@]}" \
    "${STD_NORM_ARGS[@]}" \
    ${OPTIM_OFFLOAD_ARGS[@]+"${OPTIM_OFFLOAD_ARGS[@]}"} \
    "${EVAL_ARGS[@]}" \
    "${EXTRA_TRAIN_ARGS_ARR[@]}" \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr "${LR:-1e-6}" \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT:-harbor-slime-grpo}" \
    --wandb-group "${WANDB_GROUP:-${RUN_ID}}" \
    --sglang-router-port "$SGLANG_ROUTER_PORT"
