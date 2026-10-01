#!/usr/bin/env bash
# Miles GRPO on Harbor tasks — runs on the Ray head INSIDE the Miles runtime (via
# node_entry.sh -> miles_runtime/ray_node.sh). All knobs come from the config env
# (configs/*.env), already in the environment.
#
# Layout knobs: LAYOUT colocate|disagg, TRAIN_GPUS (disagg: trainer GPUs, rest -> SGLang),
#   ACTOR_NODES (trainer nodes; disagg multi-node: whole nodes), ENGINE_TP, TP, CP, MTPG.
# Algorithm: ARM lora|full, LORA_RANK/ALPHA/TARGETS, LR, RBS (prompts/step), NS (samples/prompt),
#   NUM_ROLLOUT, ASYNC 0|1 (train_async.py --fully-async), MAX_SEQ_LEN (trajectory cap), MAXRESP (per turn).
set -euo pipefail
: "${RUN_DIR:?}" "${HARBOR_TASKS_DIR:?}" "${MILES_NUM_NODES:?}"
ARM="${ARM:-lora}"; LAYOUT="${LAYOUT:-disagg}"; ASYNC="${ASYNC:-0}"
MODEL_TYPE="${MODEL_TYPE:-qwen3.5-9B}"
HF_CKPT="${HF_CKPT:?config must set HF_CKPT (HF model dir)}"
GPUS_PER_NODE="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
TOTAL_GPUS=$(( GPUS_PER_NODE * MILES_NUM_NODES ))
TRAIN_GPUS="${TRAIN_GPUS:-4}"
TP="${TP:-2}"; CP="${CP:-1}"; MTPG="${MTPG:-16384}"; OFFLOAD="${OFFLOAD:-0}"
ENGINE_TP="${ENGINE_TP:-1}"; MEMF="${MEMF:-0.8}"
LORA_RANK="${LORA_RANK:-32}"; LORA_ALPHA="${LORA_ALPHA:-32}"; LORA_TARGETS="${LORA_TARGETS:-all-linear}"
LR="${LR:-}"; [ -n "${LR}" ] || { [ "${ARM}" = lora ] && LR=1e-5 || LR=1e-6; }
RBS="${RBS:-8}"; NS="${NS:-8}"; NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"; MAXRESP="${MAXRESP:-16384}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"; EXTRA="${EXTRA:-}"
SESSION_WORKERS="${SESSION_WORKERS:-32}"; SESSION_PORT="${SESSION_PORT:-30000}"

mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/ckpt" "${RUN_DIR}/dumps"
# ---- prompts: one row per task (metadata selects the Harbor task + agent)
DATA="${RUN_DIR}/data/train.jsonl"
# opencode: no compaction (the trajectory must stay one linear session), no sub-agents,
# no title-generation side calls.
DEFAULT_OPENCODE_CONFIG='{"compaction": {"auto": false}, "permission": {"task": "deny"}, "agent": {"title": {"disable": true}}}'
if [ ! -s "${DATA}" ]; then
    python3 "$(dirname "$0")/../tools/prepare_data.py" --tasks-dir "${HARBOR_TASKS_DIR}" --out "${DATA}" \
        ${TASK_IDS_FILE:+--ids-file "${TASK_IDS_FILE}"} --agent opencode \
        ${AGENT_IMPORT_PATH:+--agent-import-path "${AGENT_IMPORT_PATH}"} \
        --opencode-config "${OPENCODE_CONFIG:-${DEFAULT_OPENCODE_CONFIG}}"
fi

# ---- wait for every node's Harbor agent server
want="${MILES_NUM_NODES}"
for _ in $(seq 1 300); do
    have="$(wc -l < "${HARBOR_AGENT_SERVERS_FILE}" 2>/dev/null || echo 0)"
    [ "${have}" -ge "${want}" ] && break
    sleep 2
done
echo "[driver] agent servers: $(tr '\n' ' ' < "${HARBOR_AGENT_SERVERS_FILE}")"

read -ra MODEL_ARGS <<< "$(python3 /root/miles/miles/utils/external_utils/model_args_utils.py "${MODEL_TYPE}")"
args=(
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${HF_CKPT}" --megatron-to-hf-mode bridge
    --save "${RUN_DIR}/ckpt" --save-interval "${SAVE_INTERVAL}"
    --prompt-data "${DATA}" --input-key prompt --metadata-key metadata --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size "${RBS}" --n-samples-per-prompt "${NS}"
    --global-batch-size "$((RBS * NS))" --rollout-temperature 1 --rollout-top-p 1
    --rollout-max-response-len "${MAXRESP}" --max-seq-len "${MAX_SEQ_LEN}"
    --balance-data --seed "${SEED:-1234}"
    --tensor-model-parallel-size "${TP}" --sequence-parallel --pipeline-model-parallel-size 1
    --context-parallel-size "${CP}" --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MTPG}"
    --advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00
    --eps-clip 0.2 --eps-clip-high 0.28
    --optimizer adam --lr "${LR}" --lr-decay-style constant --weight-decay "${WEIGHT_DECAY:-0.0}"
    --adam-beta1 0.9 --adam-beta2 0.98
    --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash
    --num-gpus-per-node "${GPUS_PER_NODE}"
    # agentic: TITO session server + Harbor trials on per-node agent servers
    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path hm_agent.run
    --custom-rm-path hm_rollout.reward_func
    --custom-reward-post-process-path hm_rollout.post_process_rewards
    --tito-model "${TITO_MODEL:-qwen35}" --use-session-server
    # The TITO family's parsers are NOT applied to the engines automatically: without them the
    # session returns raw "</think>...<tool_call>" text and no tool_calls (agent stops after 1 turn).
    --sglang-reasoning-parser "${REASONING_PARSER:-qwen3}" --sglang-tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}"
    --session-server-port "${SESSION_PORT}" --session-server-workers "${SESSION_WORKERS}"
    --session-message-matcher "${SESSION_MATCHER:-loose_tool_call}"
    --rollout-num-gpus-per-engine "${ENGINE_TP}" --sglang-mem-fraction-static "${MEMF}"
    --sglang-context-length "${MAX_SEQ_LEN}"
)
[ "${ASYNC}" = 1 ] || args+=(--rollout-function-path hm_rollout.RolloutFn)
[ "${ENGINE_TP}" -gt 1 ] && args+=(--sglang-disable-custom-all-reduce)   # broken on hel
[ "${OFFLOAD}" = 1 ] && args+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)
if [ "${LAYOUT}" = colocate ]; then
    args+=(--colocate --actor-num-nodes "${MILES_NUM_NODES}" --actor-num-gpus-per-node "${GPUS_PER_NODE}")
else
    if [ "${TRAIN_GPUS}" -ge "${GPUS_PER_NODE}" ]; then   # whole trainer nodes
        ACTOR_NODES=$(( TRAIN_GPUS / GPUS_PER_NODE ))
        args+=(--actor-num-nodes "${ACTOR_NODES}" --actor-num-gpus-per-node "${GPUS_PER_NODE}"
               --rollout-num-gpus "$(( TOTAL_GPUS - ACTOR_NODES * GPUS_PER_NODE ))")
    else
        args+=(--actor-num-nodes 1 --actor-num-gpus-per-node "${TRAIN_GPUS}"
               --rollout-num-gpus "$(( TOTAL_GPUS - TRAIN_GPUS ))")
    fi
    args+=(--update-weight-transfer-mode broadcast)
fi
if [ "${ARM}" = lora ]; then
    args+=(--lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}" --lora-dropout 0.0
           --target-modules "${LORA_TARGETS}" --no-gradient-accumulation-fusion --sglang-max-lora-rank "${LORA_RANK}")
    [ "${LAYOUT}" = colocate ] && args+=(--lora-base-cpu-backup)
fi
[ -n "${WANDB_API_KEY:-}" ] && [ -n "${WANDB_PROJECT:-}" ] && args+=(--use-wandb --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${RUN_NAME}" --wandb-key "${WANDB_API_KEY}")
# shellcheck disable=SC2206
args+=(${EXTRA})

printf '%s\n' "${args[@]}" | grep -v -- "${WANDB_API_KEY:-__none__}" > "${RUN_DIR}/args-${SLURM_JOB_ID:-local}.txt"
echo "[driver] ${ARM}/${LAYOUT} async=${ASYNC} nodes=${MILES_NUM_NODES} TP${TP} CP${CP} mtpg ${MTPG} engineTP ${ENGINE_TP} lr ${LR}"
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 15; done ) \
    > "${RUN_DIR}/gpu-${SLURM_JOB_ID:-local}.csv" 2>/dev/null & sampler=$!
trap 'kill ${sampler} 2>/dev/null || true' EXIT
cd /root/miles
if [ "${ASYNC}" = 1 ]; then
    python3 train_async.py --fully-async "${args[@]}"
else
    python3 train.py "${args[@]}"
fi
