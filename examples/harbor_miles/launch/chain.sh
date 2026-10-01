# shellcheck shell=bash
# Cluster-side chaining, sourced by node_entry.sh on the head node right after it takes the run lock.
#
# HM_CHAIN_MAX=N (config or env; 0/unset = off) runs the same config as up to N chunks. Each chunk queues
# its successor as soon as it starts (sbatch --dependency=afterany:<this job>, same rendered sbatch script),
# so a run continues across time limits with no local watcher (laptop/VPN loss does not stop it). The
# successor resumes from RUN_DIR/ckpt like any resubmission (same RUN_NAME).
#
# A chunk does NOT queue a successor when:
#   * HM_CHAIN_INDEX >= HM_CHAIN_MAX;
#   * RUN_DIR/chain.stop exists. A chunk that starts and finds it exits without training;
#   * the run is complete (newest checkpoint >= NUM_ROLLOUT - 1). Such a chunk also exits at once;
#   * the two previous chunks both started from the checkpoint this one starts from (crash-loop guard).
# To stop a chained run: create RUN_DIR/chain.stop or cancel the pending successor (RUN_DIR/chain.log).
# chain.log: one line per chunk: "<date> job=<id> index=<i>/<max> start_iter=<n> successor=<id|->".

hm_latest_iter() {   # newest complete checkpoint iteration (LoRA bridge adapter or Megatron tracker), -1 if none
    local d it best=-1
    for d in "${RUN_DIR}"/ckpt/iter_*/adapter; do
        [ -s "${d}/adapter_megatron_rank0.pt" ] && [ -s "${d}/training_state_rank0.pt" ] || continue
        it="$(basename "$(dirname "${d}")")"; it=$((10#${it#iter_}))
        [ "${it}" -gt "${best}" ] && best="${it}"
    done
    if [ -f "${RUN_DIR}/ckpt/latest_checkpointed_iteration.txt" ]; then
        it="$(tr -dc '0-9' < "${RUN_DIR}/ckpt/latest_checkpointed_iteration.txt")"
        [ -n "${it}" ] && [ "${it}" -gt "${best}" ] && best="${it}"
    fi
    echo "${best}"
}

hm_chain() {
    local idx="${HM_CHAIN_INDEX:-1}" max="${HM_CHAIN_MAX:-0}" log="${RUN_DIR}/chain.log" it script succ="-" why=""
    [ "${max}" -gt 0 ] 2>/dev/null || return 0
    it="$(hm_latest_iter)"
    if [ -f "${RUN_DIR}/chain.stop" ]; then
        echo "$(date '+%F %T') job=${SLURM_JOB_ID} index=${idx}/${max} start_iter=${it} successor=- (chain.stop: exiting)" >> "${log}"
        hm_log "chain.stop present: not training, no successor"; exit 0
    fi
    if [ -n "${NUM_ROLLOUT:-}" ] && [ "${it}" -ge $(( NUM_ROLLOUT - 1 )) ]; then
        echo "$(date '+%F %T') job=${SLURM_JOB_ID} index=${idx}/${max} start_iter=${it} successor=- (run complete: exiting)" >> "${log}"
        hm_log "run complete (iter ${it} >= NUM_ROLLOUT-1): not training, no successor"; exit 0
    fi
    if [ "${idx}" -ge "${max}" ]; then
        why="last chunk"
    elif [ -f "${log}" ] && [ "$(tail -2 "${log}" | grep -c " start_iter=${it} ")" -ge 2 ]; then
        why="crash-loop guard: the two previous chunks also started at iter ${it}"
    else
        script="$(scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | sed -n 's/^ *Command=//p' | awk '{print $1}')"
        [ -f "${script}" ] || script="$(dirname "${SCOMPOSE_PKGS:-/nonexistent}")/sbatch.sh"
        if [ -f "${script}" ]; then
            # Clean environment: nothing of this allocation (SLURM_*, CUDA_VISIBLE_DEVICES, ...) leaks into the successor.
            succ="$(env -i HOME="${HOME}" USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-$(id -un)}" PATH="${PATH}" \
                ${SLURM_CONF:+SLURM_CONF="${SLURM_CONF}"} HM_CHAIN_INDEX=$((idx + 1)) HM_CHAIN_MAX="${max}" \
                sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" "${script}" 2>&1 | tail -1)"
            [[ "${succ}" =~ ^[0-9]+ ]] && succ="${succ%%;*}" || { why="sbatch failed: ${succ}"; succ="-"; }
        else
            why="no sbatch script found"
        fi
    fi
    echo "$(date '+%F %T') job=${SLURM_JOB_ID} index=${idx}/${max} start_iter=${it} successor=${succ}${why:+ (${why})}" >> "${log}"
    hm_log "chain: index ${idx}/${max}, start iter ${it}, successor ${succ}${why:+ (${why})}"
}
