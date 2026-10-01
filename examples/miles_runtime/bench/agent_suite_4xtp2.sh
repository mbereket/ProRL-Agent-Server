#!/usr/bin/env bash
# Rollout capacity on a FULL node as 4 x TP2 engines (the rollout phase of a 1-node colocated run, or the engine
# node of a 2-node run). agent_sim sessions sweep; set SIM_ARGS/TURNS for the context profile.
#   agent_suite_4xtp2.sh SUITE [MODEL]
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
SUITE="${1:?suite}"; MODEL="${2:-${MILES_STACK_ROOT}/models/Qwen3.5-9B}"
OUT="${MILES_STACK_ROOT}/bench/${SUITE}"; mkdir -p "${OUT}"
SESSIONS="${SESSIONS:-24 48}"; TURNS="${TURNS:-75}"; MEMF="${MEMF:-0.85}"; CTX="${CTX:-131072}"
SIM_ARGS="${SIM_ARGS:---base 8000 --tool 560 --gen 960 --ctx-cap 131072}"; EXTRA_SGL="${EXTRA_SGL:-}"
"${MR}/mrun" --no-nv -- bash "${MR}/bench/prepare.sh" > "${OUT}/prepare.log" 2>&1
BASE_PORT=$((40000 + (${SLURM_JOB_ID:-0} % 100) * 20)); pids=(); urls=""
for i in 0 1 2 3; do
    port=$((BASE_PORT + i)); urls="${urls}http://127.0.0.1:${port},"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$((2 * i)),$((2 * i + 1))" MRUN_APPTAINER_ARGS=--pid MRUN_CACHE_ROOT="/tmp/miles-${USER}-${SLURM_JOB_ID:-x}-e${i}" \
        "${MR}/mrun" -- python3 -m sglang.launch_server --model-path "${MODEL}" --host 127.0.0.1 --port "${port}" --tp 2 \
        --mem-fraction-static "${MEMF}" --trust-remote-code --disable-custom-all-reduce --context-length "${CTX}" ${EXTRA_SGL} \
        > "${OUT}/server-e${i}.log" 2>&1 &
    pids+=($!)
done
urls="${urls%,}"
for i in 0 1 2 3; do for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$((BASE_PORT + i))/health" >/dev/null && break; sleep 5; done; done
for s in ${SESSIONS}; do
    mr_log "sessions=${s}"
    # shellcheck disable=SC2086
    "${MR}/mrun" --no-nv -- python3 "${MR}/bench/agent_sim.py" --urls "${urls}" --sessions "${s}" --turns "${TURNS}" ${SIM_ARGS} \
        --name "4xTP2-m${MEMF}-s${s}" --out "${OUT}" > "${OUT}/client-s${s}.log" 2>&1
done
kill "${pids[@]}" 2>/dev/null; sleep 5; kill -9 "${pids[@]}" 2>/dev/null
cat "${OUT}/agent_sim.jsonl"
