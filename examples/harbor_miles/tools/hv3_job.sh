#!/usr/bin/env bash
# SWE-Gym Lite on Harbor/Apptainer, no training: which Harbor-native harness works, at what
# per-trial overhead, and how it replays history (TITO session matcher decision).
#   * SGLang Qwen3.5-9B from the Miles SIF (miles_runtime/mrun), DP over GPUs, 64k context,
#     qwen3 reasoning + qwen3_coder tool parsers (as in training)
#   * tools/log_proxy.py records every chat request/response
#   * this node's Harbor agent server (singularity), trials for each agent in AGENTS
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
RUN="${HM_ROOT}/runs/${JOB_NAME:-hv3}-${SLURM_JOB_ID:-local}"
mkdir -p "${RUN}"
exec > >(tee -a "${RUN}/job.log") 2>&1
TASKS="${HARBOR_TASKS_DIR:-${HM_SHARED}/tasks/swegym-lite-v3/harbor}"
export HARBOR_TASKS_DIR="${TASKS}"
AGENTS="${AGENTS:-mini-swe-agent,terminus-2}"; ATTEMPTS="${ATTEMPTS:-2}"; NTASKS="${NTASKS:-6}"
GPUS="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
MR="${MILES_RUNTIME_DIR:-${SCOMPOSE_PKGS}/miles_runtime}"
HARBOR_DIR="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_harbor.sh")"
hm_log "harbor ${HARBOR_DIR}; ${GPUS} GPUs; agents ${AGENTS}"

SNAP="${HM_SHARED}/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
bash "${MR}/mrun" -- python3 -m sglang.launch_server --model-path "${SNAP}" --served-model-name qwen35-9b \
    --host 0.0.0.0 --port 30500 --tp 1 --dp "${GPUS}" --context-length 65536 --mem-fraction-static 0.85 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --trust-remote-code > "${RUN}/sglang.log" 2>&1 &
SGL_PID=$!
"${HARBOR_DIR}/.venv/bin/python" "${HM_EXAMPLE_DIR}/tools/log_proxy.py" --upstream http://127.0.0.1:30500 \
    --port 30600 --out "${RUN}/calls.jsonl" > "${RUN}/proxy.log" 2>&1 &
for _ in $(seq 1 240); do
    curl -fs http://127.0.0.1:30500/health >/dev/null 2>&1 && break
    kill -0 "${SGL_PID}" 2>/dev/null || { tail -60 "${RUN}/sglang.log"; hm_die "sglang died"; }
    sleep 5
done
curl -fs http://127.0.0.1:30500/health >/dev/null || { tail -60 "${RUN}/sglang.log"; hm_die "sglang not healthy"; }
hm_log "sglang up"

export HM_AGENT_TIMEOUT="${HM_AGENT_TIMEOUT:-1800}"
# mini-swe-agent's litellm model_info: input budget = the 64k context.
export AGENT_MAX_INPUT_TOKENS=57344 AGENT_MAX_OUTPUT_TOKENS=8192 HARBOR_MAX_SEQ_LEN=65536
export HARBOR_AGENT_SERVERS_FILE="${RUN}/agent_servers.txt"
bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${RUN}" 18300 "${MAXC:-32}"
NODE_IP="$(hm_node_ip)"
if [ -n "${TASK_IDS_FILE:-}" ]; then
    IDS="$(awk 'NR%7==1' "${HM_EXAMPLE_DIR}/${TASK_IDS_FILE}" | head -"${NTASKS}" | tr '\n' ',')"   # spread over repos
else
    IDS="$(ls "${TASKS}" | head -"${NTASKS}" | tr '\n' ',')"
fi
hm_log "tasks: ${IDS}"
"${HARBOR_DIR}/.venv/bin/python" "${HM_EXAMPLE_DIR}/tools/validate_harbor.py" trials --tasks-dir "${TASKS}" \
    --task-ids "${IDS}" --agents "${AGENTS}" --attempts "${ATTEMPTS}" --base-url "http://${NODE_IP}:30600/v1" \
    --model openai/qwen35-9b --max-tokens 8192 --max-seq-len 65536 --timeout 3600 --out "${RUN}/trials.jsonl" || true
python3 - "${RUN}" <<'PY'
import json, sys, statistics, collections
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/trials.jsonl")]
by = collections.defaultdict(list)
for r in rows: by[r["agent"]].append(r)
for ag, rs in by.items():
    st = collections.Counter((r.get("response") or {}).get("exit_status", "NONE") for r in rs)
    rew = [(r.get("response") or {}).get("reward", 0.0) for r in rs]
    m = [(r.get("response") or {}).get("agent_metrics") or {} for r in rs]
    setup = [x.get("agent_setup_time") for x in m if x.get("agent_setup_time") is not None]
    run_t = [x.get("agent_run_time") for x in m if x.get("agent_run_time") is not None]
    print(f"{ag}: n={len(rs)} status={dict(st)} mean_reward={statistics.mean(rew):.3f} "
          f"agent_setup p50={statistics.median(setup) if setup else '-'} agent_run p50={statistics.median(run_t) if run_t else '-'} "
          f"wall p50={statistics.median(r['wall_s'] for r in rs):.0f}s")
PY
hm_log "done"
kill "${SGL_PID}" 2>/dev/null || true
