#!/usr/bin/env bash
# Miles GRPO on Harbor tasks — runs on the Ray head INSIDE the Miles runtime (via
# node_entry.sh -> miles_runtime/ray_node.sh). All knobs come from the config env
# (configs/*.env), already in the environment.
#
# Layout knobs: LAYOUT colocate|disagg, TRAIN_GPUS (disagg: trainer GPUs, rest -> SGLang),
#   ACTOR_NODES (trainer nodes; disagg multi-node: whole nodes), ENGINE_TP, TP, CP, MTPG.
# Algorithm: ARM lora|full, LORA_RANK/ALPHA/TARGETS, LORA_SERVE adapter|merged, LR, RBS (prompts/step), NS (samples/prompt),
#   NUM_ROLLOUT, ASYNC 0|1 (train_async.py --fully-async), MAX_SEQ_LEN (trajectory cap), MAXRESP (per turn).
set -euo pipefail
: "${RUN_DIR:?}" "${HARBOR_TASKS_DIR:?}" "${MILES_NUM_NODES:?}"
ARM="${ARM:-lora}"; LAYOUT="${LAYOUT:-disagg}"; ASYNC="${ASYNC:-0}"
# Colocated trainer + engines cannot overlap rollout and training: always the synchronous loop.
if [ "${LAYOUT}" = colocate ] && [ "${ASYNC}" = 1 ]; then echo "[driver] LAYOUT=colocate -> ASYNC=0 (sync loop)"; ASYNC=0; fi
# ROLLOUT_ONLY=1: rollout-only measurement (base pass rates, session lengths / overflow at a cap, trace dumps). No trainer
# (Miles --debug-rollout-only): EVERY GPU of the job serves, as TOTAL_GPUS / ENGINE_TP engines with the base weights. Sync
# loop, unfiltered groups, no eval: NUM_ROLLOUT rollouts x RBS tasks x NS attempts; concurrency = sandbox slots (dispatch
# queues the rest). Dumps on by default (ROLLOUT_DUMPS=0 turns them off): RUN_DIR/dumps/rollout_<id>.pt (token ids, loss
# masks: trainer replay) and RUN_DIR/dumps/traj_<id>.jsonl. Trials: RUN_DIR/trials-<job>.jsonl. The job ends after the last
# rollout. Leave HM_CHAIN_MAX unset.
ROLLOUT_ONLY="${ROLLOUT_ONLY:-0}"
if [ "${ROLLOUT_ONLY}" = 1 ]; then ASYNC=0; LAYOUT=disagg; DROP_ZERO_STD=0; EVAL_INTERVAL=""; fi
# EVAL_FROM_RUN=<RUN_NAME | run dir> [EVAL_STEP=<N|latest>] [EVAL_WATCH_EVERY=K]: DECOUPLED EVAL of a saved LoRA adapter.
#   The trainer (same layout as the source run: TRAIN_GPUS/TP/CP; the adapter is saved as per-rank shards) loads
#   <src>/ckpt/iter_N/adapter via --lora-adapter-path; the startup weight sync pushes W + (alpha/r)BA to plain engines (merged
#   serving, exactly like training rollouts); the eval set (TASK_IDS_FILE x EVAL_N, same harness/sampling, unfiltered) runs
#   before any training; the job stops as soon as `eval 0:` lands. Nothing is written into the source run.
#   Trials: RUN_DIR/trials-<job>.jsonl with split=eval@N; summary: RUN_DIR/eval@N-<job>.json + a row in RUN_DIR/eval_summary.tsv.
#   EVAL_WATCH_EVERY=K: keep going: evaluate each newer checkpoint >= last+K as it appears (poll 60 s) until none arrives
#   for EVAL_WATCH_IDLE_MIN (default 90) or the source run has chain.stop.
#   Fails fast if the adapter does not load (Miles would otherwise continue with a fresh B=0 adapter = the base model).
if [ -n "${EVAL_FROM_RUN:-}" ]; then
    case "${EVAL_FROM_RUN}" in /*) SRC_RUN_DIR="${EVAL_FROM_RUN}" ;; *) SRC_RUN_DIR="$(dirname "${RUN_DIR}")/${EVAL_FROM_RUN}" ;; esac
    [ -d "${SRC_RUN_DIR}/ckpt" ] || { echo "[driver] FATAL: EVAL_FROM_RUN: no ckpt dir in ${SRC_RUN_DIR}" >&2; exit 3; }
    [ "${SRC_RUN_DIR%/}" != "${RUN_DIR%/}" ] || { echo "[driver] FATAL: EVAL_FROM_RUN must not be this run (give the eval its own RUN_NAME)" >&2; exit 3; }
    EVAL_ONLY=1; EVAL_BEFORE_TRAIN=1; EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"; HM_CHAIN_MAX=0
fi
MODEL_TYPE="${MODEL_TYPE:-qwen3.5-9B}"
HF_CKPT="${HF_CKPT:?config must set HF_CKPT (HF model dir)}"
GPUS_PER_NODE="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
TOTAL_GPUS=$(( GPUS_PER_NODE * MILES_NUM_NODES ))
TRAIN_GPUS="${TRAIN_GPUS:-4}"
TP="${TP:-2}"; CP="${CP:-1}"; MTPG="${MTPG:-16384}"
ENGINE_TP="${ENGINE_TP:-2}"; MEMF="${MEMF:-0.8}"
LORA_RANK="${LORA_RANK:-32}"; LORA_ALPHA="${LORA_ALPHA:-32}"; LORA_TARGETS="${LORA_TARGETS:-all-linear}"
LR="${LR:-}"; [ -n "${LR}" ] || { [ "${ARM}" = lora ] && LR=1e-5 || LR=1e-6; }
# ARM=full is the one-flag full-FT switch; on a <=4-GPU trainer it needs CPU Adam at 64k
# (STACK §9). LoRA keeps Adam on GPU.
if [ -z "${OFFLOAD:-}" ]; then
    if [ "${ARM}" = full ] && [ "${TRAIN_GPUS}" -le 4 ]; then OFFLOAD=1; else OFFLOAD=0; fi
fi
RBS="${RBS:-8}"; NS="${NS:-8}"; NUM_ROLLOUT="${NUM_ROLLOUT:-20}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"; MAXRESP="${MAXRESP:-16384}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"; EXTRA="${EXTRA:-}"
SESSION_WORKERS="${SESSION_WORKERS:-32}"
# Session servers listen on SESSION_PORT..+SESSION_WORKERS-1. Default block 65460-65491: above dfw's ephemeral range
# (9000-65000; a fixed 30000 block collided with an outbound socket: EADDRINUSE at startup) and above Ray's 65010-65458.
# Bind-checked here; if the block is taken, the first free block below it is used.
if [ -z "${SESSION_PORT:-}" ]; then
    SESSION_PORT="$(python3 -c 'import socket, sys
n = int(sys.argv[1])
def free(b):
    for p in range(b, b + n):
        s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try: s.bind(("0.0.0.0", p))
        except OSError: return False
        finally: s.close()
    return True
print(next(b for b in [65460] + list(range(65460 - n, 40000, -n)) if free(b)))' "${SESSION_WORKERS}")"
fi

mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/ckpt" "${RUN_DIR}/dumps"
# ---- prompts: one row per task (metadata selects the Harbor task + agent)
DATA="${RUN_DIR}/data/train.jsonl"
# Harness = Harbor agent (HARNESS) + optional custom class (AGENT_IMPORT_PATH) + kwargs + AGENT_ENV (JSON, agent process env;
# e.g. BLAS/OpenMP thread caps for data-analysis tasks).
HARNESS="${HARNESS:-mini-swe-agent}"
case "${HARNESS}" in
    mini-swe-agent)
        AGENT_IMPORT_PATH="${AGENT_IMPORT_PATH:-harbor_miles_agents.mini_swe_agents:PreinstalledMiniSweAgent}"
        AGENT_KWARGS="${AGENT_KWARGS:-{\"max_tokens\": ${MAXRESP:-8192}}}" ;;
    codex)
        # Responses API -> Miles session server /v1/responses (miles_patches 0001). Path A's codex settings:
        # no sub-agents, no code mode, no web search.
        AGENT_IMPORT_PATH="${AGENT_IMPORT_PATH:-harbor_miles_agents.codex_agents:PreinstalledCodex}"
        AGENT_KWARGS="${AGENT_KWARGS:-{\"web_search\": \"disabled\", \"codex_config\": {\"features.multi_agent\": false, \"features.code_mode\": false}}}" ;;
    opencode)
        # no compaction (the trajectory must stay one linear session), no sub-agents, no title calls
        OPENCODE_CONFIG="${OPENCODE_CONFIG:-{\"compaction\": {\"auto\": false}, \"permission\": {\"task\": \"deny\"}, \"agent\": {\"title\": {\"disable\": true}}}}" ;;
esac
if [ ! -s "${DATA}" ]; then
    _tmp="${DATA}.tmp.$(hostname -s).$$"
    python3 "$(dirname "$0")/../tools/prepare_data.py" --tasks-dir "${HARBOR_TASKS_DIR}" --out "${_tmp}" \
        ${TASK_IDS_FILE:+--ids-file "${TASK_IDS_FILE}"} --agent "${HARNESS}" \
        ${AGENT_IMPORT_PATH:+--agent-import-path "${AGENT_IMPORT_PATH}"} \
        ${AGENT_KWARGS:+--agent-kwargs "${AGENT_KWARGS}"} ${OPENCODE_CONFIG:+--opencode-config "${OPENCODE_CONFIG}"} \
        ${AGENT_ENV:+--agent-env "${AGENT_ENV}"}
    mv -f "${_tmp}" "${DATA}"
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
    # harbor/* metrics in every mode's perf line + quarantine of non-finite rollout logprobs (nan_logprob_*)
    --custom-rollout-log-function-path hm_rollout.log_rollout_data
    --tito-model "${TITO_MODEL:-qwen35}" --use-session-server
    # The TITO family's parsers are NOT applied to the engines automatically: without them the
    # session returns raw "</think>...<tool_call>" text and no tool_calls (agent stops after 1 turn).
    --sglang-reasoning-parser "${REASONING_PARSER:-qwen3}" --sglang-tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}"
    --session-server-port "${SESSION_PORT}" --session-server-workers "${SESSION_WORKERS}"
    --session-message-matcher "${SESSION_MATCHER:-loose_tool_call}"
    --rollout-num-gpus-per-engine "${ENGINE_TP}" --sglang-mem-fraction-static "${MEMF}"
    --sglang-context-length "${MAX_SEQ_LEN}"
    # Old-policy logprobs = the behavior policy's own (rollout) logprobs: skips a forward pass
    # (-24% step, STACK) and is the correct ratio baseline under async staleness.
    --use-rollout-logprobs --log-probs-chunk-size "${LOGPROB_CHUNK:-4096}"
)
# In-flight session cap. Every in-flight agent session keeps its prefix in an engine's radix
# cache between turns; past the engines' KV capacity the cache thrashes (hit ratio collapses,
# decode slows 2-3x, qwen27b). Cap = engines x floor(KV_FRACTION x ENGINE_KV_TOKENS / AVG_CTX),
# and never more than the sandbox slots. ENGINE_KV_TOKENS = SGLang's max_total_num_tokens per
# engine (logged at engine start; hel TP1 @ mem 0.8 = 773763; TP2 ~ 2x).
if [ "${LAYOUT}" = colocate ] || [ "${ROLLOUT_ONLY}" = 1 ]; then N_ENGINES=$(( TOTAL_GPUS / ENGINE_TP )); else N_ENGINES=$(( (TOTAL_GPUS - TRAIN_GPUS) / ENGINE_TP )); fi
ENGINE_KV_TOKENS="${ENGINE_KV_TOKENS:-$(( ENGINE_TP == 1 ? 773763 : 913457 * ENGINE_TP ))}"   # measured on hel/dfw @ mem 0.8: TP1 773,763; TP2 1,826,914
AVG_CTX="${AVG_CTX:-$(( MAX_SEQ_LEN * 3 / 4 ))}"
KV_CAP=$(( N_ENGINES * ENGINE_KV_TOKENS * ${KV_FRACTION_PCT:-80} / 100 / AVG_CTX ))
SANDBOX_CAP=$(( ${HM_SANDBOXES_PER_NODE:-32} * MILES_NUM_NODES ))
SESSION_CAP="${SESSION_CAP:-$(( KV_CAP < SANDBOX_CAP ? KV_CAP : SANDBOX_CAP ))}"
echo "[driver] session cap ${SESSION_CAP} (${N_ENGINES} engines x TP${ENGINE_TP}, kv/engine ${ENGINE_KV_TOKENS}, avg ctx ${AVG_CTX} -> kv cap ${KV_CAP}; sandbox cap ${SANDBOX_CAP})"
if [ "${ASYNC}" = 1 ]; then
    # Fully async: the engines keep SESSION_CAP trajectories in flight across weight updates;
    # the trainer drains RBS groups per step; groups may be up to MAX_STALENESS versions old.
    args+=(--async-max-concurrent-samples "${ASYNC_CONCURRENCY:-${SESSION_CAP}}"
           --max-weight-staleness "${MAX_STALENESS:-2}" --async-unused-samples-handler retry)
else
    args+=(--rollout-function-path hm_rollout.RolloutFn)
    # Sync: one step's groups run together; oversubscribing the engines thrashes the cache too.
    [ $(( RBS * NS )) -le "${SESSION_CAP}" ] || echo "[driver] WARNING: RBS x NS = $(( RBS * NS )) > session cap ${SESSION_CAP}"
fi
[ "${DROP_ZERO_STD:-0}" = 1 ] && args+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter)
# Periodic eval on the same harness/sampling (no dynamic filter): EVAL_INTERVAL steps, EVAL_N attempts per task,
# EVAL_DATA (default: the training prompts = optimization check on the full training set).
# The default eval set is a copy of the training prompts tagged metadata.hm_split=eval, so HM_TRIAL_LOG rows tell eval
# trials from training trials (eval runs while the async producer keeps generating training groups).
# hm_eval_data <tag> <out>: the training prompts with metadata.hm_split=<tag> (built once; unique tmp + atomic rename).
hm_eval_data() {
    if [ ! -s "$2" ]; then
        local _tmp="$2.tmp.$(hostname -s).$$"
        python3 -c 'import json, sys
with open(sys.argv[2], "w") as out:
    for line in open(sys.argv[1]):
        if line.strip():
            row = json.loads(line); row.setdefault("metadata", {})["hm_split"] = sys.argv[3]; out.write(json.dumps(row) + "\n")' \
            "${DATA}" "${_tmp}" "$1"
        mv -f "${_tmp}" "$2"
    fi
}
if [ -n "${EVAL_INTERVAL:-}" ] && [ -z "${EVAL_DATA:-}" ] && [ -z "${EVAL_FROM_RUN:-}" ]; then
    EVAL_DATA="${RUN_DIR}/data/eval.jsonl"
    hm_eval_data eval "${EVAL_DATA}"
fi
if [ -n "${EVAL_INTERVAL:-}" ] && [ -z "${EVAL_FROM_RUN:-}" ]; then   # EVAL_FROM_RUN adds its eval args per checkpoint
    args+=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data "${EVAL_NAME:-train}" "${EVAL_DATA}"
           --n-samples-per-eval-prompt "${EVAL_N:-2}")
    [ "${EVAL_BEFORE_TRAIN:-0}" = 1 ] || [ "${EVAL_ONLY:-0}" = 1 ] || args+=(--skip-eval-before-train)
fi
# Resume (chained jobs, same RUN_NAME).
#  LoRA (bridge): adapter + optimizer + iteration from the newest complete ckpt/iter_N/adapter via --lora-adapter-path,
#    rollout/data-source state from ckpt/ via --load, and --start-rollout-id N+1 (miles_patches 0002 keeps both).
#  Full FT: Megatron checkpoint (latest_checkpointed_iteration.txt) via --load.
LATEST_ADAPTER=""
for d in $(ls -d "${RUN_DIR}"/ckpt/iter_*/adapter 2>/dev/null | sort -r); do
    if [ -s "${d}/adapter_megatron_rank0.pt" ] && [ -s "${d}/training_state_rank0.pt" ]; then LATEST_ADAPTER="${d}"; break; fi
done
if [ -n "${EVAL_FROM_RUN:-}" ]; then
    :   # decoupled eval: the source run's adapter is passed per checkpoint below; never resume this run's own state
elif [ "${ARM}" = lora ] && [ -n "${LATEST_ADAPTER}" ]; then
    it="$(basename "$(dirname "${LATEST_ADAPTER}")")"; it=$((10#${it#iter_}))
    args+=(--lora-adapter-path "${LATEST_ADAPTER}" --load "${RUN_DIR}/ckpt" --start-rollout-id $((it + 1)))
    echo "[driver] resuming LoRA from ${LATEST_ADAPTER} (iter ${it}), next rollout $((it + 1))"
elif [ -f "${RUN_DIR}/ckpt/latest_checkpointed_iteration.txt" ]; then
    args+=(--load "${RUN_DIR}/ckpt")
    echo "[driver] resuming from ${RUN_DIR}/ckpt (iter $(cat "${RUN_DIR}/ckpt/latest_checkpointed_iteration.txt"))"
fi
[ "${ENGINE_TP}" -gt 1 ] && args+=(--sglang-disable-custom-all-reduce)   # broken on hel
[ "${OFFLOAD}" = 1 ] && args+=(--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer)
# GatedDeltaNet context-parallel mode when CP > 1 (H2H_SPEC: headwise; never chunkwise).
[ "${CP}" -gt 1 ] && [ -n "${LINEAR_CP_MODE:-headwise}" ] && args+=(--linear-cp-mode "${LINEAR_CP_MODE:-headwise}")
if [ "${ROLLOUT_ONLY}" = 1 ]; then
    [ $(( GPUS_PER_NODE % ENGINE_TP )) -eq 0 ] \
        || { echo "[driver] FATAL: ENGINE_TP ${ENGINE_TP} does not tile a ${GPUS_PER_NODE}-GPU node" >&2; exit 2; }
    # Miles sizes the (unused) actor from --rollout-num-gpus under --debug-rollout-only; no trainer GPUs are reserved.
    args+=(--debug-rollout-only --rollout-num-gpus "${TOTAL_GPUS}")
    if [ "${ROLLOUT_DUMPS:-1}" = 1 ]; then
        args+=(--save-debug-rollout-data "${RUN_DIR}/dumps/rollout_{rollout_id}.pt"
               --save-debug-trajectory-data "${RUN_DIR}/dumps/traj_{rollout_id}.jsonl")
    fi
    echo "[driver] ROLLOUT_ONLY: ${N_ENGINES} engines x TP${ENGINE_TP} on ${TOTAL_GPUS} GPUs, ${NUM_ROLLOUT} rollout(s) x ${RBS} tasks x ${NS}, no training"
elif [ "${LAYOUT}" = colocate ]; then
    # Trainer and engines time-share every GPU (offload between phases; with LORA_SERVE=merged the patch set keys the
    # sleep/wake layout on "LoRA is trained"). Engines: all GPUs / ENGINE_TP.
    args+=(--colocate --actor-num-nodes "${MILES_NUM_NODES}" --actor-num-gpus-per-node "${GPUS_PER_NODE}")
else
    # Split layout. Trainer = TRAIN_GPUS on the first node (< 1 node) or whole nodes; engines = every other GPU of the
    # job, on any node (e.g. 4 trainer + 12 engine GPUs on 2 nodes, 8 + 24 on 4). Engines never straddle nodes.
    if [ "${TRAIN_GPUS}" -lt "${GPUS_PER_NODE}" ]; then
        [ $(( (GPUS_PER_NODE - TRAIN_GPUS) % ENGINE_TP )) -eq 0 ] \
            || { echo "[driver] FATAL: ENGINE_TP ${ENGINE_TP} does not tile the ${GPUS_PER_NODE}-${TRAIN_GPUS} engine GPUs left on the trainer node" >&2; exit 2; }
    else
        [ $(( TRAIN_GPUS % GPUS_PER_NODE )) -eq 0 ] \
            || { echo "[driver] FATAL: TRAIN_GPUS ${TRAIN_GPUS} must be < ${GPUS_PER_NODE} or a multiple of it" >&2; exit 2; }
    fi
    [ $(( TOTAL_GPUS - TRAIN_GPUS )) -gt 0 ] || { echo "[driver] FATAL: no GPUs left for engines" >&2; exit 2; }
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
           --target-modules "${LORA_TARGETS}" --no-gradient-accumulation-fusion)
    if [ "${LORA_SERVE:-adapter}" = merged ]; then
        # Train LoRA, serve merged (STACK §10, patch set miles_runtime/patches/lora-serve-merged, added by
        # node_entry): each sync pushes W + (alpha/r)BA as full weights; SGLang runs a plain model (no LoRA
        # kernels, no lora_path), ~26% faster rollout. No LoRA serving flags.
        args+=(--lora-serve-merged)
    else
        args+=(--sglang-max-lora-rank "${LORA_RANK}")
        [ "${LAYOUT}" = colocate ] && args+=(--lora-base-cpu-backup)
        # csgmv (default) LoRA serving costs ~25% decode; triton is what STACK validated with TP>1 engines.
        args+=(--sglang-lora-backend "${LORA_BACKEND:-triton}")
    fi
fi
# W&B reads WANDB_API_KEY from the environment (never on the command line: args are logged to RUN_DIR/args-*.txt).
[ -n "${WANDB_API_KEY:-}" ] && [ -n "${WANDB_PROJECT:-}" ] && args+=(--use-wandb --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${RUN_NAME}")
# shellcheck disable=SC2206
args+=(${EXTRA})

printf '%s\n' "${args[@]}" | grep -v -- "${WANDB_API_KEY:-__none__}" > "${RUN_DIR}/args-${SLURM_JOB_ID:-local}.txt"
echo "[driver] ${ARM}/${LAYOUT} async=${ASYNC} nodes=${MILES_NUM_NODES} TP${TP} CP${CP} mtpg ${MTPG} engineTP ${ENGINE_TP} lr ${LR}"
( while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 15; done ) \
    > "${RUN_DIR}/gpu-${SLURM_JOB_ID:-local}.csv" 2>/dev/null & sampler=$!
trap 'kill ${sampler} 2>/dev/null || true' EXIT
cd /root/miles
if [ "${ASYNC}" = 1 ]; then train_cmd=(python3 train_async.py --fully-async "${args[@]}"); else train_cmd=(python3 train.py "${args[@]}"); fi
# stop_after_eval <status file>: pass the run's log through; stop the run as soon as `eval 0:` lands (status ok). With an adapter
# to evaluate, fail fast if it did not load: Miles only warns and would evaluate a fresh B=0 adapter, i.e. the base model.
stop_after_eval() {
    local line st="$1"
    echo running > "${st}"
    while IFS= read -r line; do
        printf '%s\n' "${line}"
        case "${line}" in
            *"adapter weights could not be loaded"*)
                echo "[driver] FATAL: the LoRA adapter did not load; stopping (this would have evaluated the base model)"
                echo adapter_load_failed > "${st}"; pkill -TERM -f "train_async.py|train.py" || true ;;
            *"Successfully loaded LoRA adapter from"*)
                echo "[driver] adapter loaded: ${line##*from }" ;;
            *" - eval 0: {"*)
                echo "[driver] EVAL_ONLY: eval landed; stopping"
                [ "$(cat "${st}")" = running ] && echo ok > "${st}"
                pkill -TERM -f "train_async.py|train.py" || true ;;
        esac
    done
}
# hm_eval_summary <tag> <source> <status>: per-task + overall results of the trials tagged <tag> (overlong scored per
# HM_OVERLONG_REWARD) -> RUN_DIR/<tag>-<job>.json and one row in RUN_DIR/eval_summary.tsv; printed to the log too.
hm_eval_summary() {
    python3 - "${HM_TRIAL_LOG:-${RUN_DIR}/trials-${SLURM_JOB_ID:-local}.jsonl}" "$1" "$2" "$3" "${RUN_DIR}" "${SLURM_JOB_ID:-local}" \
        "${HM_OVERLONG_REWARD:-verifier}" <<'PY'
import collections, json, os, statistics, sys, time
log, tag, source, status, run_dir, job, ol_policy = sys.argv[1:8]
rows = [json.loads(l) for l in open(log) if l.strip().startswith("{")] if os.path.exists(log) else []
rows = [r for r in rows if r.get("split") == tag]
ok = [r for r in rows if not r.get("infra_failure")]
overlong = lambda r: r.get("exit_status") == "SequenceLengthLimitExceeded"
score = lambda r: 0.0 if (ol_policy == "zero" and overlong(r)) else float(r.get("reward") or 0.0)
per = collections.defaultdict(list)
for r in ok:
    per[r["instance_id"]].append(r)
tasks = {k: {"attempts": len(v), "successes": round(sum(score(r) for r in v), 3),
             "overlong": sum(map(overlong, v)), "wall_p50_s": round(statistics.median(r["wall_s"] for r in v))}
         for k, v in sorted(per.items())}
summ = {"tag": tag, "source": source, "status": status, "job": job, "time": time.strftime("%F %T"),
        "trials": len(rows), "infra_failures": len(rows) - len(ok), "tasks": len(tasks),
        "pass": round(statistics.mean(score(r) for r in ok), 4) if ok else None,
        "raw_verifier_pass": round(statistics.mean(float(r.get("reward") or 0) for r in ok), 4) if ok else None,
        "overlong_rate": round(sum(map(overlong, ok)) / len(ok), 4) if ok else None,
        "task_mean_pass": round(statistics.mean(t["successes"] / t["attempts"] for t in tasks.values()), 4) if tasks else None,
        "per_task": tasks}
json.dump(summ, open(os.path.join(run_dir, f"{tag}-{job}.json"), "w"), indent=1)
tsv = os.path.join(run_dir, "eval_summary.tsv")
new = not os.path.exists(tsv)
with open(tsv, "a") as f:
    if new:
        f.write("time\tjob\ttag\tstatus\ttrials\tinfra\ttasks\tpass\ttask_mean_pass\traw_verifier_pass\toverlong_rate\tsource\n")
    f.write("\t".join(str(summ[k]) for k in ("time", "job", "tag", "status", "trials", "infra_failures", "tasks", "pass",
                                             "task_mean_pass", "raw_verifier_pass", "overlong_rate", "source")) + "\n")
print(f"[driver] eval summary {tag}: " + json.dumps({k: v for k, v in summ.items() if k != "per_task"}))
PY
}
if [ -n "${EVAL_FROM_RUN:-}" ]; then
    # Decoupled eval of saved adapters (see EVAL_FROM_RUN at the top).
    hm_iters() { ls -d "${SRC_RUN_DIR}"/ckpt/iter_*/adapter 2>/dev/null | sed -n 's#.*/iter_0*\([0-9][0-9]*\)/adapter$#\1#p' | sort -n; }
    hm_ckpt_ready() {   # every trainer rank's adapter + training-state shard written, and quiet for >= 60 s
        local d="$1" r n
        [ -s "${d}/adapter_megatron_rank0.pt" ] || return 1
        n="$(ls "${d}"/adapter_megatron_rank*.pt 2>/dev/null | wc -l)"
        for r in $(seq 0 $((n - 1))); do [ -s "${d}/adapter_megatron_rank${r}.pt" ] && [ -s "${d}/training_state_rank${r}.pt" ] || return 1; done
        [ $(( $(date +%s) - $(stat -c %Y "${d}/adapter_megatron_rank$((n - 1)).pt") )) -ge 60 ]
    }
    hm_wait_gpus_free() {   # engines/trainer of the previous eval released (this node); up to 5 min
        local _i
        for _i in $(seq 1 60); do
            [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" -eq 0 ] && return 0
            sleep 5
        done
        echo "[driver] WARNING: GPUs still busy 5 min after the previous eval"
    }
    want="${EVAL_STEP:-latest}"; every="${EVAL_WATCH_EVERY:-0}"; last=-1; idle_since="$(date +%s)"
    echo "[driver] EVAL_FROM_RUN ${SRC_RUN_DIR}: step ${want}, watch every ${every}, ${EVAL_N:-2} attempts x tasks of ${TASK_IDS_FILE:-all}"
    while true; do
        pick=""
        for n in $(hm_iters); do
            hm_ckpt_ready "${SRC_RUN_DIR}/ckpt/iter_$(printf %07d "${n}")/adapter" || continue
            if [ "${last}" -ge 0 ]; then [ "${n}" -ge $(( last + every )) ] && pick="${n}"
            elif [ "${want}" = latest ] || [ "${n}" -eq "${want}" ] 2>/dev/null; then pick="${n}"; fi
        done
        if [ -z "${pick}" ]; then
            [ "${last}" -ge 0 ] || [ "${every}" -gt 0 ] || { echo "[driver] FATAL: no complete checkpoint ${want} in ${SRC_RUN_DIR}/ckpt" >&2; exit 3; }
            [ "${every}" -gt 0 ] || break
            if [ -f "${SRC_RUN_DIR}/chain.stop" ] || [ $(( $(date +%s) - idle_since )) -ge $(( ${EVAL_WATCH_IDLE_MIN:-90} * 60 )) ]; then
                echo "[driver] EVAL watch: no newer checkpoint (chain.stop or idle ${EVAL_WATCH_IDLE_MIN:-90} min); done"; break
            fi
            sleep 60; continue
        fi
        adapter="${SRC_RUN_DIR}/ckpt/iter_$(printf %07d "${pick}")/adapter"
        nsh="$(ls "${adapter}"/adapter_megatron_rank*.pt | wc -l)"
        [ "${nsh}" -eq "${TRAIN_GPUS}" ] || { echo "[driver] FATAL: ${adapter} has ${nsh} rank shards but TRAIN_GPUS=${TRAIN_GPUS}: use the source run's trainer layout (TRAIN_GPUS/TP/CP)" >&2; exit 3; }
        tag="eval@${pick}"; efile="${RUN_DIR}/data/${tag}.jsonl"; hm_eval_data "${tag}" "${efile}"
        st="${RUN_DIR}/.eval-status-${SLURM_JOB_ID:-local}-${pick}"
        eargs=(--eval-interval "${EVAL_INTERVAL}" --eval-prompt-data "${EVAL_NAME:-train}" "${efile}"
               --n-samples-per-eval-prompt "${EVAL_N:-2}" --lora-adapter-path "${adapter}")
        printf '%s\n' "# ${tag}" "${eargs[@]}" >> "${RUN_DIR}/args-${SLURM_JOB_ID:-local}.txt"
        echo "[driver] EVAL_FROM_RUN: ${tag} <- ${adapter}"
        "${train_cmd[@]}" "${eargs[@]}" 2>&1 | stop_after_eval "${st}" || true
        status="$(cat "${st}" 2>/dev/null || echo missing)"
        hm_eval_summary "${tag}" "${adapter}" "${status}" || true
        [ "${status}" = ok ] || { echo "[driver] FATAL: eval of ${adapter} ended with status ${status}" >&2; exit 3; }
        last="${pick}"; idle_since="$(date +%s)"
        [ "${every}" -gt 0 ] || break
        hm_wait_gpus_free
    done
    exit 0
fi
if [ "${EVAL_ONLY:-0}" = 1 ]; then
    # Rollout-only measurement (base pass rates, session lengths, overflow at a cap): the eval set (EVAL_DATA x EVAL_N,
    # same harness/sampling, unfiltered) runs before any training; the run stops as soon as its metrics line lands.
    # Trials are in trials-<job>.jsonl (split=eval). Needs EVAL_BEFORE_TRAIN=1 and EVAL_INTERVAL set (the config's job).
    echo "[driver] EVAL_ONLY: eval before train, then stop"
    "${train_cmd[@]}" 2>&1 | stop_after_eval "${RUN_DIR}/.eval-status-${SLURM_JOB_ID:-local}" || true
    hm_eval_summary eval "${HF_CKPT}" "$(cat "${RUN_DIR}/.eval-status-${SLURM_JOB_ID:-local}" 2>/dev/null || echo missing)" || true
    exit 0
fi
"${train_cmd[@]}"
