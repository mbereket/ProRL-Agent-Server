#!/usr/bin/env bash
# Harbor-on-Apptainer validation job (1 node, 1 GPU), no Miles:
#   1. build the patched Harbor venv + in-sandbox server runtime
#   2. serve Qwen3.5-9B with SGLang (borrowed polar-slime stack) behind a logging proxy
#   3. start this node's Harbor agent server (singularity backend)
#   4. nop wave (sandbox start/stop at concurrency), real bbh-mcp opencode trials,
#      and a flush test (cancel running trials, check no leaked sandbox processes)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
RUN="${HM_ROOT}/runs/${JOB_NAME:-hv}-${SLURM_JOB_ID:-local}"
mkdir -p "${RUN}"
exec > >(tee -a "${RUN}/job.log") 2>&1
hm_log "run dir ${RUN}"
NOP_N="${NOP_N:-32}"; TRIAL_IDS="${TRIAL_IDS:-}"; ATTEMPTS="${ATTEMPTS:-1}"; FLUSH_N="${FLUSH_N:-2}"
TASKS="${HARBOR_TASKS_DIR:-${HM_SHARED}/tasks/bbh-mcp-train-v1/harbor}"
export HARBOR_TASKS_DIR="${TASKS}"

hm_log "node: $(nproc) cpus, $(free -g | awk '/Mem:/{print $2}') GB; /tmp: $(df -h /tmp | tail -1)"
df -h /tmp /dev/shm /raid /local /scratch 2>/dev/null || true
hm_log "NVINF_API_KEY $([ -n "${NVINF_API_KEY:-}" ] && echo set || echo UNSET)"
"${HM_APPTAINER}/bin/apptainer" --version
cat /proc/sys/user/max_user_namespaces || true

HARBOR_DIR="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_harbor.sh")"
SERVER_PY="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_sandbox_runtime.sh")"
hm_log "harbor ${HARBOR_DIR}; sandbox python ${SERVER_PY}"

# ---- model server (borrowed stack), logging proxy
(
    source "${HERE}/slime_sglang_env.sh"
    exec "${VIRTUAL_ENV}/bin/python" -m sglang.launch_server --model-path "${QWEN35_9B_SNAPSHOT}" --served-model-name qwen35-9b \
        --host 0.0.0.0 --port 30500 --tp 1 --context-length 131072 --mem-fraction-static 0.85 \
        --tool-call-parser qwen3_coder --reasoning-parser qwen3 --trust-remote-code
) > "${RUN}/sglang.log" 2>&1 &
SGL_PID=$!
"${HARBOR_DIR}/.venv/bin/python" "${HM_EXAMPLE_DIR}/tools/log_proxy.py" --upstream http://127.0.0.1:30500 \
    --port 30600 --out "${RUN}/calls.jsonl" > "${RUN}/proxy.log" 2>&1 &
for i in $(seq 1 180); do
    curl -fs http://127.0.0.1:30500/health >/dev/null 2>&1 && break
    kill -0 "${SGL_PID}" 2>/dev/null || { tail -60 "${RUN}/sglang.log"; hm_die "sglang died"; }
    sleep 5
done
curl -fs http://127.0.0.1:30500/health >/dev/null || { tail -60 "${RUN}/sglang.log"; hm_die "sglang not healthy"; }
hm_log "sglang up"

# ---- agent server on this node
export HM_AGENT_TIMEOUT="${HM_AGENT_TIMEOUT:-3600}"
bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${RUN}" 18300 "${MAXC:-40}"
export HARBOR_AGENT_SERVERS_FILE="${RUN}/agent_servers.txt"
NODE_IP="$(hm_node_ip)"
V=("${HARBOR_DIR}/.venv/bin/python" "${HM_EXAMPLE_DIR}/tools/validate_harbor.py")

snapshot_procs() { ps -u "${USER}" -o pid,ppid,pgid,etimes,args | grep -E 'starter|apptainer|singularity|_hbexec|opencode|faked' | grep -v grep || true; }

# ---- 1) nop wave: sandbox start/stop overhead at concurrency
hm_log "nop wave n=${NOP_N}"
"${V[@]}" nop --tasks-dir "${TASKS}" --task-ids "${NOP_IDS:-$(ls "${TASKS}" | head -1)}" --n "${NOP_N}" \
    --out "${RUN}/nop.jsonl" || hm_log "nop wave failed"
hm_log "leftover sandbox processes after nop wave:"; snapshot_procs

# ---- 2) real opencode trials (+ 3) flush test in parallel)
IDS="${TRIAL_IDS:-$(ls "${TASKS}" | tr '\n' ',' )}"
hm_log "trials: ${IDS} x ${ATTEMPTS}"
"${V[@]}" trials --tasks-dir "${TASKS}" --task-ids "${IDS}" --attempts "${ATTEMPTS}" \
    --base-url "http://${NODE_IP}:30600/v1" --model openai/qwen35-9b --out "${RUN}/trials.jsonl" &
TR_PID=$!
if [ "${FLUSH_N}" -gt 0 ]; then
    sleep 60
    hm_log "flush test: ${FLUSH_N} trials, flush_all after 240 s"
    # NOTE flush_all cancels EVERY in-flight trial on the server, including the real
    # ones above; run it on a second server so the trial wave is unaffected.
    FLUSH_RUN="${RUN}/flushsrv"; mkdir -p "${FLUSH_RUN}"
    HARBOR_AGENT_SERVERS_FILE="${FLUSH_RUN}/agent_servers.txt" bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${FLUSH_RUN}" 18301 8
    HARBOR_AGENT_SERVERS_FILE="${FLUSH_RUN}/agent_servers.txt" "${V[@]}" flush --tasks-dir "${TASKS}" \
        --task-ids "$(ls "${TASKS}" | head -1)" --n "${FLUSH_N}" --after 240 \
        --base-url "http://${NODE_IP}:30600/v1" --model openai/qwen35-9b --out "${RUN}/flush.jsonl" || true
    sleep 15
    hm_log "processes of the flush server's trials after flush_all (expect none under ${FLUSH_RUN}):"
    ps -u "${USER}" -o pid,pgid,etimes,args | grep -F "${FLUSH_RUN}" | grep -v grep || echo "none"
fi
wait "${TR_PID}" || hm_log "trial wave exited non-zero"
hm_log "leftover sandbox processes at end:"; snapshot_procs
python3 - "${RUN}" <<'PY'
import json, sys, statistics, glob, os
run = sys.argv[1]
def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for name in ("nop", "trials", "flush"):
    rows = load(f"{run}/{name}.jsonl")
    if not rows: continue
    st = {}
    for r in rows:
        s = (r.get("response") or {}).get("exit_status", "NONE"); st[s] = st.get(s, 0) + 1
    rew = [(r.get("response") or {}).get("reward", 0.0) for r in rows]
    wall = [r["wall_s"] for r in rows]
    print(f"{name}: n={len(rows)} status={st} mean_reward={statistics.mean(rew):.3f} wall p50={statistics.median(wall):.0f}s max={max(wall):.0f}s")
starts = []
for f in glob.glob(f"{run}/**/sandbox_startup.json", recursive=True):
    try: starts.append(json.load(open(f))["elapsed_sec"])
    except Exception: pass
if starts:
    starts.sort()
    print(f"sandbox startup: n={len(starts)} p50={starts[len(starts)//2]:.1f}s p90={starts[int(len(starts)*.9)]:.1f}s max={starts[-1]:.1f}s")
PY
hm_log "done"
kill "${SGL_PID}" 2>/dev/null || true
