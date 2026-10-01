#!/usr/bin/env bash
# SGLang serving-config sweep on ONE node: one server per GPU group, all configs measured concurrently.
#
#   serve_suite.sh SUITE CONFIGS_FILE [MODEL] [ADAPTER_DIR]
#
# CONFIGS_FILE lines: `<name> <tp> <lora:0|1> [extra sglang server args...]` (# comments). GPUs are handed out
# in file order (tp GPUs each); configs that do not fit wait for the next wave. Results:
# $MILES_STACK_ROOT/bench/<SUITE>/serve.jsonl (+ per-config server logs).
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
SUITE="${1:?suite}"; CFG="${2:?configs}"; MODEL="${3:-${MILES_STACK_ROOT}/models/Qwen3.5-9B}"
ADAPTER="${4:-${MILES_STACK_ROOT}/models/lora-q35-9b-r32}"
[ -f "${CFG}" ] || CFG="${MR}/bench/${CFG}"
OUT="${MILES_STACK_ROOT}/bench/${SUITE}"; mkdir -p "${OUT}"
SHAPES="${SHAPES:-2048x2048x64,32768x1024x16}"
NG="$(nvidia-smi --list-gpus | wc -l)"
BASE_PORT=$((40000 + (${SLURM_JOB_ID:-0} % 100) * 20))

# Stage the r32 adapter (a PEFT dir saved by a Miles LoRA run) once.
if [ ! -f "${ADAPTER}/adapter_config.json" ]; then
    : "${ADAPTER_SRC:?set ADAPTER_SRC to a PEFT adapter dir to stage ${ADAPTER}}"
    mkdir -p "${ADAPTER}.tmp" && cp "${ADAPTER_SRC}/adapter_config.json" "${ADAPTER_SRC}/adapter_model.safetensors" "${ADAPTER}.tmp/" \
        && mv "${ADAPTER}.tmp" "${ADAPTER}"
fi

mapfile -t lines < <(grep -v '^\s*#' "${CFG}" | grep -v '^\s*$')
i=0
while [ "${i}" -lt "${#lines[@]}" ]; do
    gpu=0; pids=(); names=()
    while [ "${i}" -lt "${#lines[@]}" ]; do
        read -r name tp lora extra <<< "${lines[$i]}"
        [ $((gpu + tp)) -le "${NG}" ] || break
        devs="$(seq -s, "${gpu}" $((gpu + tp - 1)))"; port=$((BASE_PORT + i))
        largs=(); [ "${lora}" = 1 ] && largs=(--enable-lora --lora-paths "miles_lora=${ADAPTER}" --max-lora-rank 32 --max-loras-per-batch 1)
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="${devs}" MRUN_APPTAINER_ARGS=--pid MRUN_CACHE_ROOT="/tmp/miles-${USER}-${SLURM_JOB_ID:-x}-${name}" "${MR}/mrun" -- \
            python3 -m sglang.launch_server --model-path "${MODEL}" --host 127.0.0.1 --port "${port}" --tp "${tp}" \
            --mem-fraction-static 0.8 --trust-remote-code --disable-custom-all-reduce "${largs[@]}" ${extra} \
            > "${OUT}/server-${name}.log" 2>&1 &
        pids+=($!); names+=("${name}:${port}:${lora}")
        gpu=$((gpu + tp)); i=$((i + 1))
    done
    mr_log "wave: ${names[*]}"
    clients=()
    for nl in "${names[@]}"; do
        IFS=: read -r name port lora <<< "${nl}"
        (
            for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:${port}/health" >/dev/null && break; sleep 5; done
            curl -sf "http://127.0.0.1:${port}/health" >/dev/null || { echo "${name}: server never became healthy" >&2; exit 1; }
            lr=""; [ "${lora}" = 1 ] && lr=miles_lora
            "${MR}/mrun" --no-nv -- python3 "${MR}/bench/serve_bench.py" --url "http://127.0.0.1:${port}" --name "${name}" \
                --out "${OUT}" --lora "${lr}" --shapes "${SHAPES}"
        ) > "${OUT}/client-${name}.log" 2>&1 &
        clients+=($!)
    done
    wait "${clients[@]}"
    kill "${pids[@]}" 2>/dev/null; sleep 5; kill -9 "${pids[@]}" 2>/dev/null; wait 2>/dev/null
done
mr_log "serve suite ${SUITE} done"; cat "${OUT}/serve.jsonl"
