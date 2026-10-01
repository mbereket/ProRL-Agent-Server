#!/usr/bin/env bash
# Qwen3.5-9B GRPO benchmark driver (runs on the Ray head, inside the runtime; see bench/run.sh).
# All knobs are env vars; every run writes $OUT/{args.txt,train.log,gpu_mem.csv,metrics.jsonl}.
#
#   ARM         full | lora
#   LAYOUT      colocate | disagg            (disagg: TRAIN_GPUS for Megatron, the rest for SGLang)
#   MODE        e2e | train_only             (train_only replays REPLAY=<dir>/{rollout_id}.pt, no SGLang)
#   LORA_ROLLOUT 1 | 0                       (0: --lora-train-only, rollouts on the frozen base = negative control)
set -euo pipefail
: "${MILES_STACK_ROOT:?}" "${OUT:?}"
ARM="${ARM:-lora}"; LAYOUT="${LAYOUT:-colocate}"; MODE="${MODE:-e2e}"
MODEL="${MODEL:-Qwen3.5-9B}"; MODEL_TYPE="${MODEL_TYPE:-qwen3.5-9B}"
GPUS="${GPUS:-8}"; TRAIN_GPUS="${TRAIN_GPUS:-4}"
NUM_ROLLOUT="${NUM_ROLLOUT:-6}"; RBS="${RBS:-32}"; NS="${NS:-8}"; MAXRESP="${MAXRESP:-8192}"
TP="${TP:-2}"; CP="${CP:-1}"; MTPG="${MTPG:-9216}"; OFFLOAD="${OFFLOAD:-0}"
ENGINE_TP="${ENGINE_TP:-1}"; MEMF="${MEMF:-0.6}"
LORA_RANK="${LORA_RANK:-32}"; LORA_ALPHA="${LORA_ALPHA:-32}"; LORA_TARGETS="${LORA_TARGETS:-all-linear}"
LR="${LR:-}"; [ -n "${LR}" ] || { [ "${ARM}" = lora ] && LR=1e-5 || LR=1e-6; }
SAVE_ROLLOUTS="${SAVE_ROLLOUTS:-1}"; EXTRA="${EXTRA:-}"
MODELS="${MILES_STACK_ROOT}/models"; DATA="${MILES_STACK_ROOT}/datasets"
mkdir -p "${OUT}"

read -ra MODEL_ARGS <<< "$(python3 /root/miles/miles/utils/external_utils/model_args_utils.py "${MODEL_TYPE}")"
args=(
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "${MODELS}/${MODEL}" --megatron-to-hf-mode bridge
    --prompt-data "${DATA}/dapo-math-17k/dapo-math-17k.jsonl" --input-key prompt --label-key label
    --apply-chat-template --rollout-shuffle --rm-type deepscaler
    --num-rollout "${NUM_ROLLOUT}" --rollout-batch-size "${RBS}" --n-samples-per-prompt "${NS}"
    --rollout-max-response-len "${MAXRESP}" --rollout-temperature 1 --global-batch-size "$((RBS * NS / ${STEPS:-1}))"
    --balance-data --seed 1234 --rollout-seed 1234
    --tensor-model-parallel-size "${TP}" --sequence-parallel --pipeline-model-parallel-size 1
    --context-parallel-size "${CP}" --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
    --use-dynamic-batch-size --max-tokens-per-gpu "${MTPG}"
    --advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl --entropy-coef 0.00
    --eps-clip 0.2 --eps-clip-high 0.28
    --optimizer adam --lr "${LR}" --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
    --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32 --attention-backend flash
    --num-gpus-per-node "${GPUS}" --actor-num-nodes 1
)
case "${RECOMPUTE:-full}" in
    full) args+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
    selective) args+=(--recompute-granularity selective) ;;
    none) ;;
    *) echo "bad RECOMPUTE=${RECOMPUTE}" >&2; exit 2 ;;
esac
[ "${ROLLOUT_LOGPROBS:-0}" = 1 ] && args+=(--use-rollout-logprobs)   # skip the separate old-logprob forward pass
[ "${OFFLOAD}" = 1 ] && args+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)

if [ "${MODE}" = train_only ]; then
    : "${REPLAY:?MODE=train_only needs REPLAY=<dir with {rollout_id}.pt>}"
    args+=(--debug-train-only --load-debug-rollout-data "${REPLAY}/{rollout_id}.pt" --actor-num-gpus-per-node "${GPUS}")
else
    args+=(--rollout-num-gpus-per-engine "${ENGINE_TP}" --sglang-mem-fraction-static "${MEMF}")
    [ "${ENGINE_TP}" -gt 1 ] && args+=(--sglang-disable-custom-all-reduce)   # broken on hel (see STACK.md)
    if [ "${LAYOUT}" = colocate ]; then
        args+=(--colocate --actor-num-gpus-per-node "${GPUS}")
        # trainer and engines both stay resident (no sleep/wake): needs a small trainer (LoRA) + lower MEMF
        [ "${NOOFF:-0}" = 1 ] && args+=(--no-offload-train --no-offload-rollout)
    else
        args+=(--actor-num-gpus-per-node "${TRAIN_GPUS}" --rollout-num-gpus "$((GPUS - TRAIN_GPUS))"
               --update-weight-transfer-mode broadcast)
    fi
    [ "${SAVE_ROLLOUTS}" = 1 ] && args+=(--save-debug-rollout-data "${OUT}/rollout_data/{rollout_id}.pt")
fi

if [ "${ARM}" = lora ]; then
    args+=(--lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}" --lora-dropout 0.0
           --target-modules "${LORA_TARGETS}" --no-gradient-accumulation-fusion)
    if [ "${LORA_ROLLOUT:-1}" = 0 ]; then
        args+=(--lora-train-only)   # negative control: SGLang stays on the frozen base
    elif [ "${LORA_SERVE:-adapter}" = merged ]; then
        args+=(--lora-serve-merged)  # needs patches/lora-serve-merged: plain engine, merged full-weight sync
    elif [ "${MODE}" = e2e ]; then
        args+=(--sglang-max-lora-rank "${LORA_RANK}")
        # Colocated LoRA must keep the SGLang base weights in host RAM, or rollouts after the first
        # offload run on a stale/garbled base (the 35B-A3B recipe measured KL ~1.0 without it).
        [ "${LAYOUT}" = colocate ] && args+=(--lora-base-cpu-backup)
    fi
fi
# shellcheck disable=SC2206
args+=(${EXTRA})

printf '%s\n' "${args[@]}" > "${OUT}/args.txt"
env | grep -E '^(ASYNC|NOOFF|LORA_SERVE|STEPS|ARM|LAYOUT|MODE|TP|CP|MTPG|OFFLOAD|RECOMPUTE|ROLLOUT_LOGPROBS|ENGINE_TP|MEMF|LORA_|LR|NUM_ROLLOUT|RBS|NS|MAXRESP|TRAIN_GPUS|REPLAY)=' | sort > "${OUT}/knobs.txt" || true
echo "[grpo] ${ARM}/${LAYOUT}/${MODE} TP${TP} CP${CP} mtpg ${MTPG} offload ${OFFLOAD} engineTP ${ENGINE_TP} -> ${OUT}"

# GPU memory sampler (peak per GPU, all processes on the node).
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 5; done ) \
    > "${OUT}/gpu_mem.csv" 2>/dev/null & sampler=$!
trap 'kill ${sampler} 2>/dev/null || true' EXIT

cd /root/miles
entry=train.py
if [ "${ASYNC:-0}" = 1 ]; then entry=train_async.py; args+=(--fully-async); fi   # needs LAYOUT=disagg
t0=${SECONDS}
set +e
python3 "${entry}" "${args[@]}" 2>&1 | tee "${OUT}/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "[grpo] exit ${rc} after $((SECONDS - t0)) s"
python3 "$(dirname "$0")/parse_metrics.py" "${OUT}" || true
exit "${rc}"
