#!/usr/bin/env bash
# Diagnose Ray bring-up inside the runtime; copies Ray session logs to $OUT.
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
OUT="${MILES_STACK_ROOT}/diag/ray-${SLURM_JOB_ID}"; mkdir -p "${OUT}"
for pidns in "" "--pid"; do
  tag="pidns${pidns:+-on}"; export RAY_TMPDIR="/tmp/ray-diag-${SLURM_JOB_ID}-${tag}"; mkdir -p "${RAY_TMPDIR}"
  echo "===== ${tag}"
  MRUN_APPTAINER_ARGS="${pidns}" "${MR}/mrun" -- bash -c '
    pip list 2>/dev/null | grep -i -E "^(opentelemetry|ray|grpcio|protobuf) " 
    python3 -c "import opentelemetry.sdk.metrics" 2>&1 | tail -1
    python3 -c "import ray, os; p=os.path.dirname(ray.__file__); print(p)"
    grep -rn "enable_open_telemetry\|RAY_enable_open_telemetry" $(python3 -c "import ray,os;print(os.path.dirname(ray.__file__))")/_private/ray_constants.py | head -5
    t0=$SECONDS
    timeout 300 ray start --head --include-dashboard=false --num-gpus 1 --disable-usage-stats --port 6399
    echo "ray start rc=$? after $((SECONDS-t0))s"
    ray status 2>&1 | head -15
    s=$(ls -d $RAY_TMPDIR/ray/session_* 2>/dev/null | tail -1)
    ls $s/logs | head -50
    for f in raylet.out raylet.err gcs_server.err dashboard_agent.log runtime_env_agent.log python-core-driver*; do [ -f "$s/logs/$f" ] && { echo "--- $f"; tail -25 "$s/logs/$f"; }; done
    cp -r $s/logs '"${OUT}"'/logs-'"${tag}"' 2>/dev/null
    ray stop --force >/dev/null 2>&1
  '
done
