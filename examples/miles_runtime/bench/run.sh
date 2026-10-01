#!/usr/bin/env bash
# Benchmark job entry (one srun task per node). Runs the arms of ARMS_FILE sequentially,
# each in a fresh Ray cluster (ray_node.sh), writing to $MILES_STACK_ROOT/bench/<SUITE>/<arm>/.
#
#   run.sh SUITE ARMS_FILE [PATCH_DIR]
#
# ARMS_FILE: one arm per line, `<name> KEY=VALUE ...` (knobs of bench/grpo.sh; `#` comments).
# An arm with a DONE marker (written on rc 0) is skipped, so a resubmitted job resumes the suite.
# `REPLAY=@<arm>` points a train_only arm at another arm's rollout dump.
# `MODEL_PATH=@<path>` is relative to the cluster's miles root (the parent of MILES_STACK_ROOT), e.g. @models/Qwen3.8-27B.
# `!synth <name> <seq_len> <samples> <rollouts>` writes fixed-length synthetic dumps to <name>/rollout_data.
# `!compose <name> <dir> <steps> [--max-samples N]` merges real dumps <dir>/*.pt into <steps> bigger replay steps
# (compose_replay.py); <dir> may be @<path> relative to the cluster's miles root.
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
SUITE="${1:?suite}"; ARMS_FILE="${2:?arms file}"; PATCH_DIR="${3:-}"
[ -f "${ARMS_FILE}" ] || ARMS_FILE="${MR}/bench/${ARMS_FILE}"
ROOT="${MILES_STACK_ROOT}/bench/${SUITE}"
mkdir -p "${ROOT}"
# PATCH_DIR: one or more patch-set dirs, colon-separated, absolute or relative to miles_runtime/
patch_opts=()
IFS=: read -r -a _pdirs <<< "${PATCH_DIR}"
for d in "${_pdirs[@]}"; do
    [ -n "${d}" ] || continue
    [ -d "${d}" ] || d="${MR}/${d}"
    patch_opts+=(--patches "${d}")
done

if [ "${SLURM_NODEID:-0}" = 0 ]; then
    "${MR}/mrun" --no-nv -- bash "${MR}/bench/prepare.sh" 2>&1 | tail -20
fi

while read -r name rest; do
    [ -z "${name}" ] || [ "${name:0:1}" = "#" ] && continue
    if [ "${name}" = "!compose" ]; then   # !compose <name> <dir with real dumps *.pt> <steps>
        read -r cname csrc csteps cextra <<< "${rest}"
        [ "${csrc}" = TRACES_DIR ] && csrc="${MILES_STACK_ROOT}/traces/swegym-smoke2"   # cluster-local copy
        [[ "${csrc}" == @* ]] && csrc="$(dirname "${MILES_STACK_ROOT}")/${csrc#@}"
        if [ ! -s "${ROOT}/${cname}/rollout_data/$((csteps - 1)).pt" ]; then
            "${MR}/mrun" --no-nv -- bash -c "python3 '${MR}/bench/compose_replay.py' '${ROOT}/${cname}/rollout_data' ${csrc}/*.pt --steps ${csteps} ${cextra:-}" 2>&1 | tail -6
        fi
        continue
    fi
    if [ "${name}" = "!synth" ]; then
        read -r sname slen ssamples srollouts <<< "${rest}"
        if [ ! -s "${ROOT}/${sname}/rollout_data/$((srollouts - 1)).pt" ]; then
            "${MR}/mrun" --no-nv -- python3 "${MR}/bench/make_synthetic.py" "${ROOT}/${sname}/rollout_data" \
                --seq-len "${slen}" --samples "${ssamples}" --rollouts "${srollouts}" 2>&1 | tail -2
        fi
        continue
    fi
    out="${ROOT}/${name}"
    if [ -f "${out}/DONE" ]; then
        mr_log "arm ${name}: done already, skipping"; continue
    fi
    mkdir -p "${out}"
    kv=(); for x in ${rest}; do
        if [[ "${x}" == REPLAY=@* ]]; then x="REPLAY=${ROOT}/${x#REPLAY=@}/rollout_data"; fi
        if [[ "${x}" == MODEL_PATH=@* ]]; then x="MODEL_PATH=$(dirname "${MILES_STACK_ROOT}")/${x#MODEL_PATH=@}"; fi
        kv+=("${x}")
    done
    mr_log "arm ${name}: ${kv[*]}"
    t0=${SECONDS}
    arm_i=$((${arm_i:-0} + 1))
    # fresh Ray port block per arm: a previous arm's stragglers must not collide with the next head
    env "${kv[@]}" OUT="${out}" RAY_PORT_BASE="$((65010 + (${SLURM_JOB_ID:-0} % 12) * 40 + (arm_i % 4) * 10))" \
        bash "${MR}/ray_node.sh" "${patch_opts[@]}" -- bash "${MR}/bench/grpo.sh" > "${out}/job.log" 2>&1
    rc=$?
    if [ "${rc}" = 0 ]; then date -u +%FT%TZ > "${out}/DONE"; fi
    mr_log "arm ${name}: rc ${rc} in $((SECONDS - t0)) s ($(tail -c 300 "${out}/job.log" | tr '\n' ' ' | cut -c1-200))"
    sleep 10
done < "${ARMS_FILE}"
mr_log "suite ${SUITE}: all arms attempted"
