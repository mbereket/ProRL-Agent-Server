#!/usr/bin/env bash
# Rollout-side A100-vs-H100 comparison for Qwen3.5-9B: three SGLang engines side by side on one node
# (TP1 on GPU0, TP2 on GPUs 1-2, TP4 on GPUs 4-7), each driven by the agentic multi-turn simulator
# (sim_multiturn.py, copied from the qwen27b agent) at SWE-Gym shape: ~830 decode + ~585 tool tokens/turn.
# Concurrency per engine = engine capacity / context (1.0x: the non-thrashing operating point).
#   serve9b.sh            (env: SIM_LIMIT seconds per sim, default 600; CTXS default "32768 65536")
set -uo pipefail
: "${MILES_STACK_ROOT:?}" "${MILES_OWNER_ROOT:?}"
MRUN="${SCOMPOSE_PKGS}/miles_runtime/mrun"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-${MILES_STACK_ROOT}/models/Qwen3.5-9B}"
RESULTS="${MILES_OWNER_ROOT}/results/serve9b-${CLUSTER_ALIAS:-x}-${SLURM_JOB_ID:-nojob}"
export TMPDIR="/tmp/fleet-${SLURM_JOB_ID:-nojob}"   # SGLang ZMQ ipc sockets: keep paths short + node-local
mkdir -p "${RESULTS}" "${TMPDIR}"
TURN_OUT="${TURN_OUT:-830}"; TOOL="${TOOL:-585}"; SIM_LIMIT="${SIM_LIMIT:-600}"; CTXS="${CTXS:-32768 65536}"
CAR=(); [ "${CLUSTER_ALIAS:-}" = hel ] && CAR=(--disable-custom-all-reduce)
log() { echo "[serve9b $(date +%H:%M:%S) $(hostname -s)] $*"; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "${RESULTS}/gpus.csv"
log "results ${RESULTS}"

launch() { # name gpus tp port
    CUDA_VISIBLE_DEVICES=$2 "${MRUN}" python -m sglang.launch_server --model-path "${MODEL}" --tp "$3" \
        --host 127.0.0.1 --port "$4" --trust-remote-code --enable-metrics --context-length 70000 \
        --mem-fraction-static "${MEMF:-0.85}" "${CAR[@]}" ${SGLANG_EXTRA:-} > "${RESULTS}/$1.server.log" 2>&1 &
    eval "PID_$1=$!"
}
healthy() { # name port
    local s; s=$(date +%s)
    until curl -sf "http://127.0.0.1:$2/health_generate" >/dev/null 2>&1; do
        local pv="PID_$1"; kill -0 "${!pv}" 2>/dev/null || { log "$1 DIED"; tail -60 "${RESULTS}/$1.server.log"; return 1; }
        [ $(( $(date +%s) - s )) -gt 1800 ] && { log "$1 timeout"; return 1; }
        sleep 10
    done
    log "$1 healthy after $(( $(date +%s) - s )) s"
}
capacity() { # name -> "tokens maxreq"
    local f="${RESULTS}/$1.server.log"
    echo "$(grep -oE 'max_total_num_tokens=[0-9]+' "$f" | tail -1 | cut -d= -f2) $(grep -oE 'max_running_requests=[0-9]+' "$f" | tail -1 | cut -d= -f2)"
}
sim() { # name port label maxctx agents prompt limit
    "${MRUN}" python "${HERE}/sim_multiturn.py" --url "http://127.0.0.1:$2" --agents "$5" --max-ctx "$4" \
        --prompt-len "$6" --turn-out "${TURN_OUT}" --tool-len "${TOOL}" --time-limit "$7" \
        --label "$1/$3" --out "${RESULTS}/$1.$3.json" > "${RESULTS}/$1.$3.log" 2>&1 || log "sim $1 $3 failed"
    echo "$1 $3 agents=$5 $(grep -E '"(decode_tok_s|cached_ratio|running_reqs_mean|n_errors)"' "${RESULTS}/$1.$3.log" | tr -d '\n ')"
}
lane() { # name port
    read -r tok req < <(capacity "$1")
    echo "$1 capacity tokens=${tok} max_running=${req}" | tee -a "${RESULTS}/capacity.txt"
    local ctx a
    for ctx in ${CTXS}; do
        a=$(( tok / ctx )); [ "${req:-0}" -gt 0 ] && [ "$a" -gt "$req" ] && a=$req; [ "$a" -gt 256 ] && a=256; [ "$a" -lt 1 ] && continue
        sim "$1" "$2" "c$((ctx / 1024))k_a$a" "$ctx" "$a" $(( ctx - 5 * (TURN_OUT + TOOL) )) "${SIM_LIMIT}"
    done
    # SWE-Gym-shaped trajectories: 8k prompt -> 40k (~22 turns), concurrency sized at the mean context (24k)
    a=$(( tok / 24576 )); [ "${req:-0}" -gt 0 ] && [ "$a" -gt "$req" ] && a=$req; [ "$a" -gt 256 ] && a=256
    sim "$1" "$2" "traj40k_a$a" 40960 "$a" 8192 $(( SIM_LIMIT * 2 ))
}

launch tp1 0 1 30010
launch tp2 1,2 2 30020
launch tp4 4,5,6,7 4 30040
ok=()
for n in tp1:30010 tp2:30020 tp4:30040; do healthy "${n%%:*}" "${n##*:}" && ok+=("$n"); done
pids=()
for n in "${ok[@]}"; do lane "${n%%:*}" "${n##*:}" > "${RESULTS}/lane_${n%%:*}.txt" 2>&1 & pids+=($!); done
wait "${pids[@]}"
for f in "${RESULTS}"/lane_*.txt; do echo "== $f"; cat "$f"; done
pkill -f sglang.launch_server 2>/dev/null; sleep 5
log done
