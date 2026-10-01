#!/usr/bin/env bash
# hel: terminus-2 on SWE-Gym tasks with staged SIFs (harness validation + replay behavior), while
# staging SIFs for the dfw-staged SWE-Gym set in the background.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
mkdir -p "${HM_ROOT}/runs/stage-${SLURM_JOB_ID}"
bash "${HERE}/stage_sifs.sh" "${HM_EXAMPLE_DIR}/configs/swegym-dfw-staged93-refs.txt" "${STAGE_JOBS:-6}" \
    > "${HM_ROOT}/runs/stage-${SLURM_JOB_ID}/stage.log" 2>&1 &
export AGENTS="${AGENTS:-terminus-2}" TASK_IDS_FILE="${TASK_IDS_FILE:-configs/swegym-hel-staged.txt}"
bash "${HERE}/hv3_job.sh" || true
wait
