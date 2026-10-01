#!/usr/bin/env bash
# Rollout-side capacity for the 1-node agentic shape: 4 GPUs of engines (as in a 4 train + 4 rollout run).
# GPUs 0-3: 4 x TP1 engines; GPUs 4-7: 2 x TP2 engines (run concurrently, independent groups).
# Each group is driven by agent_sim.py at increasing in-flight session counts (SESSIONS, per 4-GPU group).
#   agent_suite.sh SUITE [MODEL]
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
SUITE="${1:?suite}"; MODEL="${2:-${MILES_STACK_ROOT}/models/Qwen3.5-9B}"
OUT="${MILES_STACK_ROOT}/bench/${SUITE}"; mkdir -p "${OUT}"
SESSIONS="${SESSIONS:-48 96 160}"; TURNS="${TURNS:-16}"; MEMF="${MEMF:-0.85}"
EXTRA_SGL="${EXTRA_SGL:-}"
"${MR}/mrun" --no-nv -- bash "${MR}/bench/prepare.sh" > "${OUT}/prepare.log" 2>&1
BASE_PORT=$((40000 + (${SLURM_JOB_ID:-0} % 100) * 20))
pids=()
start() {  # name devs tp port
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$2" MRUN_APPTAINER_ARGS=--pid MRUN_CACHE_ROOT="/tmp/miles-${USER}-${SLURM_JOB_ID:-x}-$1" "${MR}/mrun" -- \
        python3 -m sglang.launch_server --model-path "${MODEL}" --host 127.0.0.1 --port "$4" --tp "$3" \
        --mem-fraction-static "${MEMF}" --trust-remote-code --disable-custom-all-reduce --context-length 65536 ${EXTRA_SGL} \
        > "${OUT}/server-$1.log" 2>&1 &
    pids+=($!)
}
for i in 0 1 2 3; do start "tp1-$i" "${i}" 1 $((BASE_PORT + i)); done
start tp2-0 4,5 2 $((BASE_PORT + 4)); start tp2-1 6,7 2 $((BASE_PORT + 5))
for p in $(seq 0 5); do
    for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$((BASE_PORT + p))/health" >/dev/null && break; sleep 5; done
done
tp1_urls="$(for i in 0 1 2 3; do printf 'http://127.0.0.1:%d,' $((BASE_PORT + i)); done)"; tp1_urls="${tp1_urls%,}"
tp2_urls="http://127.0.0.1:$((BASE_PORT + 4)),http://127.0.0.1:$((BASE_PORT + 5))"
for s in ${SESSIONS}; do
    mr_log "sessions=${s}"
    "${MR}/mrun" --no-nv -- python3 "${MR}/bench/agent_sim.py" --urls "${tp1_urls}" --sessions "${s}" --turns "${TURNS}" \
        --name "4xTP1-s${s}" --out "${OUT}" > "${OUT}/client-4xTP1-s${s}.log" 2>&1 &
    c1=$!
    "${MR}/mrun" --no-nv -- python3 "${MR}/bench/agent_sim.py" --urls "${tp2_urls}" --sessions "${s}" --turns "${TURNS}" \
        --name "2xTP2-s${s}" --out "${OUT}" > "${OUT}/client-2xTP2-s${s}.log" 2>&1 &
    c2=$!
    wait "${c1}" "${c2}"
done
kill "${pids[@]}" 2>/dev/null; sleep 5; kill -9 "${pids[@]}" 2>/dev/null
cat "${OUT}/agent_sim.jsonl"
