#!/usr/bin/env bash
# Mirror a harbor_miles run's small files and its job logs locally (input for the analysis tools; read-only on the cluster).
#
# usage: fetch_run.sh CLUSTER RUN_NAME_OR_ABS_DIR [LOCAL_DIR (default ./runs-mirror/<cluster>/<run>)]
#
# 1. resolves the remote run dir like hmruns.py (miles/runs/<RUN> first, then legacy path-b / qwen27b / shared/hm roots)
# 2. pulls trials-*.jsonl, args-*.txt, chain.log, jobs.log, gpu-*.csv, node-*.csv, agent_servers-*.txt (top level only;
#    no trials/ dirs, checkpoints or dumps) into LOCAL_DIR/
# 3. pulls the job logs of the run's job ids (from those file names and chain.log / jobs.log) into LOCAL_DIR/joblogs/
#    (main <jobid>-hm-<name>[.<ts>].log and per-step <jobid>.<step>-<name>.log)
# Re-run to refresh (rsync pulls only what changed). Uses cluster-tools scrsync.py (pull-only rsync) and its venv python.
# Note: scrsync.py refuses paths containing credential-like words (e.g. "token", "secret").
set -euo pipefail

CT="${CLUSTER_TOOLS:-$HOME/Desktop/code/cluster-tools}"
PY="${CT}/.venv/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 2 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
CL="$1"
RUN="$2"

REMOTE="$("${PY}" "${HERE}/hmruns.py" resolve "${CL}:${RUN}")"
NAME="$(basename "${REMOTE}")"
LOCAL="${3:-./runs-mirror/${CL}/${NAME}}"
mkdir -p "${LOCAL}/joblogs"
LOCAL="$(cd "${LOCAL}" && pwd)"
echo "remote ${CL}:${REMOTE} -> ${LOCAL}"

( cd "${CT}" && "${PY}" scrsync.py "${CL}" "${REMOTE}/" "${LOCAL}/" \
    --include='trials-*.jsonl' --include='args-*.txt' --include='chain.log' --include='jobs.log' \
    --include='gpu-*.csv' --include='node-*.csv' --include='agent_servers-*.txt' --exclude='*' )

JOBS="$("${PY}" "${HERE}/hmruns.py" jobids "${LOCAL}")"
if [ -z "${JOBS}" ]; then
    echo "no job ids found in ${LOCAL} (no trials-/args-/gpu-<jobid> files or chain.log): no job logs fetched"
    exit 0
fi
echo "job ids: $(echo ${JOBS})"

# Remote job-log files of those ids, grouped by directory -> one scrsync per directory with exact-name includes.
# shellcheck disable=SC2086
LOGS="$("${PY}" "${HERE}/hmruns.py" joblogs "${CL}:${REMOTE}" ${JOBS})"
if [ -z "${LOGS}" ]; then
    echo "no job logs found for job ids $(echo ${JOBS}) in the job-log dirs"
    exit 0
fi
for DIR in $(printf '%s\n' "${LOGS}" | xargs -n1 dirname | sort -u); do
    INCLUDES=()
    while IFS= read -r F; do
        [ "$(dirname "${F}")" = "${DIR}" ] && INCLUDES+=("--include=$(basename "${F}")")
    done <<< "${LOGS}"
    ( cd "${CT}" && "${PY}" scrsync.py "${CL}" "${DIR}/" "${LOCAL}/joblogs/" "${INCLUDES[@]}" --exclude='*' )
done
echo "done: ${LOCAL} ($(ls "${LOCAL}/joblogs" | wc -l | tr -d ' ') job logs)"
