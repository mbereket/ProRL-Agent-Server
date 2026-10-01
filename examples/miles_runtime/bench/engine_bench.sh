#!/usr/bin/env bash
# Engine capacity per engine TP (layout study part 3). For each TP in TPS: fill the node with 8/TP engines, record
# KV tokens / mamba slots / max running requests from the server logs, then drive agent_sim at increasing in-flight
# sessions PER GPU (the knee shows up as cache-hit collapse + latency blow-up).
#   engine_bench.sh SUITE [MODEL]
# env: TPS="1 2 4 8"  PER_GPU="12 24 48"  CAP=65536  MEMF=0.85
#      SHAPE="--base 3000 --gen 830 --tool 585 --turns-mean 17 --turns 400"   (SWE-Gym codex, qwen27b calibration)
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
SUITE="${1:?suite}"; MODEL="${2:-${MILES_STACK_ROOT}/models/Qwen3.5-9B}"
OUT="${MILES_STACK_ROOT}/bench/${SUITE}"; mkdir -p "${OUT}"
TPS="${TPS:-1 2 4 8}"; PER_GPU="${PER_GPU:-12 24 48}"; CAP="${CAP:-65536}"; MEMF="${MEMF:-0.85}"
SHAPE="${SHAPE:---base 3000 --gen 830 --tool 585 --turns-mean 17 --turns 400}"; EXTRA_SGL="${EXTRA_SGL:-}"
NG="$(nvidia-smi --list-gpus | wc -l)"
[ "${MODEL}" = "${MILES_STACK_ROOT}/models/Qwen3.5-9B" ] && "${MR}/mrun" --no-nv -- bash "${MR}/bench/prepare.sh" > "${OUT}/prepare.log" 2>&1
BASE_PORT=$((40000 + (${SLURM_JOB_ID:-0} % 100) * 20))
for tp in ${TPS}; do
    ne=$((NG / tp)); pids=(); urls=""
    for i in $(seq 0 $((ne - 1))); do
        port=$((BASE_PORT + i)); urls="${urls}http://127.0.0.1:${port},"
        devs="$(seq -s, $((i * tp)) $((i * tp + tp - 1)))"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${devs}" MRUN_APPTAINER_ARGS=--pid MRUN_CACHE_ROOT="/tmp/miles-${USER}-${SLURM_JOB_ID:-x}-tp${tp}e${i}" \
            "${MR}/mrun" -- python3 -m sglang.launch_server --model-path "${MODEL}" --host 127.0.0.1 --port "${port}" --tp "${tp}" \
            --mem-fraction-static "${MEMF}" --trust-remote-code --disable-custom-all-reduce --context-length "${CAP}" ${EXTRA_SGL} \
            > "${OUT}/server-tp${tp}-e${i}.log" 2>&1 &
        pids+=($!)
    done
    urls="${urls%,}"
    for i in $(seq 0 $((ne - 1))); do for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$((BASE_PORT + i))/health" >/dev/null && break; sleep 5; done; done
    log0="${OUT}/server-tp${tp}-e0.log"
    kv=$(grep -oE '#tokens: [0-9]+' "${log0}" | head -1 | grep -oE '[0-9]+'); mamba=$(grep -oE 'max_mamba_cache_size: [0-9]+' "${log0}" | head -1 | grep -oE '[0-9]+')
    maxrun=$(grep -oE 'capped to [0-9]+ by the mamba' "${log0}" | head -1 | grep -oE '[0-9]+')
    echo "{\"name\": \"cap-tp${tp}\", \"tp\": ${tp}, \"engines\": ${ne}, \"kv_tokens_per_engine\": ${kv:-null}, \"mamba_slots_per_engine\": ${mamba:-null}, \"max_running_per_engine\": ${maxrun:-null}, \"memf\": ${MEMF}, \"cap\": ${CAP}}" >> "${OUT}/capacity.jsonl"
    for pg in ${PER_GPU}; do
        s=$((pg * NG)); mr_log "TP${tp}: ${ne} engines, ${s} sessions (${pg}/GPU)"
        # shellcheck disable=SC2086
        "${MR}/mrun" --no-nv -- python3 "${MR}/bench/agent_sim.py" --urls "${urls}" --sessions "${s}" --gpus "${NG}" ${SHAPE} \
            --ctx-cap "${CAP}" --name "tp${tp}-cap${CAP}-pg${pg}" --out "${OUT}" > "${OUT}/client-tp${tp}-pg${pg}.log" 2>&1
    done
    kill "${pids[@]}" 2>/dev/null; sleep 8; kill -9 "${pids[@]}" 2>/dev/null; wait 2>/dev/null; sleep 5
done
cat "${OUT}/capacity.jsonl" "${OUT}/agent_sim.jsonl"
