#!/usr/bin/env bash
# Sandbox teardown test (no GPU/model): nop waves and a flush of sleeping sandboxes;
# after each, list any leftover apptainer processes (starter, squashfuse_ll,
# fuse-overlayfs, faked, in-container server) with their parents.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
RUN="${HM_ROOT}/runs/${JOB_NAME:-hv2}-${SLURM_JOB_ID:-local}"
mkdir -p "${RUN}"
exec > >(tee -a "${RUN}/job.log") 2>&1
TASKS="${HARBOR_TASKS_DIR:-${HM_SHARED}/tasks/bbh-mcp-train-v1/harbor}"
export HARBOR_TASKS_DIR="${TASKS}"
HARBOR_DIR="$(bash "${HM_EXAMPLE_DIR}/setup/ensure_harbor.sh")"
export HARBOR_AGENT_SERVERS_FILE="${RUN}/agent_servers.txt"
bash "${HM_EXAMPLE_DIR}/launch/start_agent_server.sh" "${RUN}" 18300 "${MAXC:-64}"
V=("${HARBOR_DIR}/.venv/bin/python" "${HM_EXAMPLE_DIR}/tools/validate_harbor.py")
ID1="$(ls "${TASKS}" | head -1)"
leftovers() {
    sleep 10
    local n
    n="$( (ps -u "${USER}" -o pid=,args= | grep -E 'libexec/starter|squashfuse_ll|fuse-overlayfs|faked|_hbexec' | grep -v grep || true) | wc -l)"
    hm_log "leftover sandbox processes after $1: ${n}"
    ps -u "${USER}" -o pid,ppid,pgid,etimes,comm | grep -E 'starter|squashfuse|fuse-overlay|faked|python3' | grep -v grep | head -20 || true
    (ls -d /local/hm-overlay-${USER}-${SLURM_JOB_ID}/* /tmp/hm-overlay-${USER}-${SLURM_JOB_ID}/* 2>/dev/null || true) | wc -l | xargs echo "overlay dirs left:"
}
hm_log "nop wave 1 (n=${NOP_N:-64})"
"${V[@]}" nop --tasks-dir "${TASKS}" --task-ids "${ID1}" --n "${NOP_N:-64}" --out "${RUN}/nop1.jsonl" | tail -3
leftovers "nop wave 1"
hm_log "sleep sandboxes + flush_all after 60 s (n=${FLUSH_N:-16})"
"${V[@]}" sleepflush --tasks-dir "${TASKS}" --task-ids "${ID1}" --n "${FLUSH_N:-16}" --after 60 --out "${RUN}/sleepflush.jsonl" | tail -3
leftovers "flush"
python3 - "${RUN}" <<'PY'
import json, sys, glob, statistics
run = sys.argv[1]
for name in ("nop1", "sleepflush"):
    rows = [json.loads(l) for l in open(f"{run}/{name}.jsonl")]
    st = {}
    for r in rows:
        s = (r.get("response") or {}).get("exit_status", "NONE"); st[s] = st.get(s, 0) + 1
    print(name, len(rows), st, "wall p50", statistics.median(r["wall_s"] for r in rows))
starts = sorted(json.load(open(f))["elapsed_sec"] for f in glob.glob(f"{run}/trials/**/sandbox_startup.json", recursive=True))
if starts:
    print(f"sandbox startup n={len(starts)} p50={starts[len(starts)//2]:.1f}s p90={starts[int(len(starts)*.9)]:.1f}s max={starts[-1]:.1f}s")
PY
hm_log "done"
