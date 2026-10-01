#!/usr/bin/env bash
# FLEET: make a cluster without a polar-slime work root (aws-iad, ord, draco) ready for harbor_miles runs.
# Idempotent and resumable; run as a 1-node CPU-heavy job (CLUSTER set by submit.py):
#   tasks   SWE-Gym Lite task packages (configs/fleet/swegym-lite-v3.tgz, from dfw) -> ${HM_SHARED}/tasks/
#   uv      ${HM_UV}
#   model   Qwen/Qwen3.5-9B @ c2022362 -> ${HM_SHARED}/hf_home/hub (same snapshot path the configs use)
#   tools   Harbor checkout+venv, sandbox runtime, codex toolchain (setup/ensure_*.sh, under ${HM_ROOT})
#   sifs    task images -> ${HM_SHARED}/harbor_sif_images/<Harbor cache name>.sif, rand48 first, then the rest of
#           the 100-task pool; STAGE_JOBS parallel pulls with apptainer tmp + layer cache on node-local disk
#   fleet_stage.sh [steps...]   (default: tasks uv model tools sifs)
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
F="${HM_EXAMPLE_DIR}/configs/fleet"
REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
SIFDIR="${HM_SHARED}/harbor_sif_images"
mkdir -p "${HM_SHARED}"/{tasks,bin,hf_home} "${SIFDIR}"
step_tasks() {
    if [ -f "${HM_SHARED}/tasks/swegym-lite-v3/harbor/getmoto__moto-5865/task.toml" ]; then hm_log "tasks: present"; return; fi
    tar -xzf "${F}/swegym-lite-v3.tgz" -C "${HM_SHARED}/tasks/" && hm_log "tasks: $(ls "${HM_SHARED}/tasks/swegym-lite-v3/harbor" | wc -l) task dirs"
}
step_uv() {
    if [ -x "${HM_UV}" ]; then hm_log "uv: $("${HM_UV}" --version)"; return; fi
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$(dirname "${HM_UV}")" INSTALLER_NO_MODIFY_PATH=1 sh >&2
    hm_log "uv: $("${HM_UV}" --version)"
}
step_model() {
    local d="${HM_SHARED}/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/${REV}"
    if [ -f "${d}/config.json" ]; then hm_log "model: present ${d}"; return; fi
    HF_HUB_ENABLE_HF_TRANSFER=0 "${HM_UV}" run --no-project --with huggingface_hub python -c \
        "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-9B', revision='${REV}', cache_dir='${HM_SHARED}/hf_home/hub'))" >&2
    [ -f "${d}/config.json" ] && hm_log "model: ${d} ($(du -sh "${d}/" | cut -f1))" || hm_log "model: FAILED"
}
step_tools() {
    local t0=${SECONDS}
    bash "${HM_EXAMPLE_DIR}/setup/ensure_harbor.sh" >/dev/null && hm_log "tools: harbor ok" || hm_log "tools: harbor FAILED"
    bash "${HM_EXAMPLE_DIR}/setup/ensure_sandbox_runtime.sh" >/dev/null && hm_log "tools: sandbox runtime ok" || hm_log "tools: sandbox runtime FAILED"
    bash "${HM_EXAMPLE_DIR}/setup/ensure_agent_tools.sh" codex >/dev/null && hm_log "tools: codex ok" || hm_log "tools: codex FAILED"
    hm_log "tools: done in $((SECONDS - t0)) s"
}
pick_local() {   # largest node-local disk with >= 200 GB free; else /dev/shm (RAM, charged to the job's memory); else lustre
    local best="" bestfree=0 d f
    for d in /raid /local /scratch /opt/dlami/nvme /mnt/local /tmp; do
        mkdir -p "${d}/fleet-${USER}-${SLURM_JOB_ID:-x}" 2>/dev/null || continue
        f="$(df -BG --output=avail "${d}" 2>/dev/null | tail -1 | tr -dc 0-9)"
        echo "[pick_local] ${d}: ${f:-?}G free" >&2
        [ "${f:-0}" -gt "${bestfree}" ] && { best="${d}/fleet-${USER}-${SLURM_JOB_ID:-x}"; bestfree="${f}"; }
    done
    if [ "${bestfree}" -ge 200 ]; then echo "${best}"; return; fi
    f="$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9)"
    echo "[pick_local] /dev/shm: ${f:-?}G free" >&2
    if [ "${f:-0}" -ge 300 ] && mkdir -p "/dev/shm/fleet-${USER}-${SLURM_JOB_ID:-x}"; then echo "/dev/shm/fleet-${USER}-${SLURM_JOB_ID:-x}"; return; fi
    echo "${HM_ROOT}/cache/sif-build-${SLURM_JOB_ID:-x}"
}
pull_one() {
    local ref="$1" image name
    image="${ref}"; case "${ref##*/}" in *:*) ;; *) image="${ref}:latest" ;; esac
    name="$(echo "${image}" | tr '/:' '__').sif"
    [ -s "${SIFDIR}/${name}" ] && { echo "present ${name}"; return 0; }
    local t0=${SECONDS} part="${SIFDIR}/.${name}.part.$$"
    if "${HM_APPTAINER}/bin/apptainer" pull -F "${part}" "docker://${image}" >"${LOCAL}/${name}.log" 2>&1; then
        mv -f "${part}" "${SIFDIR}/${name}"; echo "pulled ${name} in $((SECONDS - t0)) s ($(du -h "${SIFDIR}/${name}" | cut -f1))"
    else
        echo "FAILED ${image}: $(tail -2 "${LOCAL}/${name}.log" | tr '\n' ' ' | cut -c1-300)"
    fi
}
step_sifs() {
    LOCAL="$(pick_local)"; mkdir -p "${LOCAL}"
    export LOCAL SIFDIR HM_APPTAINER
    export APPTAINER_TMPDIR="${LOCAL}/tmp" APPTAINER_CACHEDIR="${LOCAL}/cache"
    mkdir -p "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
    local jobs="${STAGE_JOBS:-16}" ncpu; ncpu="$(nproc)"
    export APPTAINER_MKSQUASHFS_PROCS="$(( ncpu / jobs > 1 ? ncpu / jobs : 1 ))"
    hm_log "sifs: ${jobs} parallel pulls, ${APPTAINER_MKSQUASHFS_PROCS} mksquashfs procs each, build dir ${LOCAL} ($(df -h --output=avail "${LOCAL}" | tail -1) free), -> ${SIFDIR}"
    export -f pull_one
    local list t0
    for list in swegym-rand48-s0.refs pool-rest.refs; do
        t0=${SECONDS}
        grep -v '^\s*$' "${F}/${list}" | xargs -P "${jobs}" -I{} bash -c 'pull_one "$@"' _ {}
        hm_log "sifs: ${list} pass done in $((SECONDS - t0)) s"
    done
    # Retry pass at low parallelism: concurrent first pulls of images sharing base layers occasionally fail in
    # apptainer's layer cache ("conveyor failed to get: unexpected end of JSON"); present SIFs are skipped.
    t0=${SECONDS}
    cat "${F}/swegym-rand48-s0.refs" "${F}/pool-rest.refs" | grep -v '^\s*$' | xargs -P "${STAGE_RETRY_JOBS:-4}" -I{} bash -c 'pull_one "$@"' _ {} | grep -v '^present '
    hm_log "sifs: retry pass done in $((SECONDS - t0)) s"
    case "${LOCAL}" in /dev/shm/*) rm -rf "${LOCAL}" ;; esac   # RAM-backed scratch: release it
}
summary() {
    local list n have name
    for list in swegym-rand48-s0.refs pool-rest.refs; do
        n=0; have=0
        while read -r ref; do
            [ -z "${ref}" ] && continue; n=$((n + 1))
            case "${ref##*/}" in *:*) name="${ref}" ;; *) name="${ref}:latest" ;; esac
            [ -s "${SIFDIR}/$(echo "${name}" | tr '/:' '__').sif" ] && have=$((have + 1))
        done < "${F}/${list}"
        echo "${list}: ${have}/${n} SIFs"
    done
    echo "tasks: $(ls "${HM_SHARED}/tasks/swegym-lite-v3/harbor" 2>/dev/null | wc -l)"
    echo "model: $(ls "${HM_SHARED}/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/" 2>/dev/null)"
    echo "harbor: $(ls -d "${HM_ROOT}"/harbor/*/ 2>/dev/null | tr '\n' ' ')"
    echo "codex: $(cat "${HM_ROOT}"/agent-tools/codex-*/VERSION 2>/dev/null)"
    echo "sandbox runtime: $(ls "${HM_ROOT}"/sandbox-runtime/ 2>/dev/null)"
}
hm_log "fleet_stage on ${CLUSTER} $(hostname -s): $(nproc) cpus; HM_SHARED=${HM_SHARED} HM_ROOT=${HM_ROOT}"
steps=("$@"); [ "${#steps[@]}" -eq 0 ] && steps=(tasks uv model tools sifs)
for s in "${steps[@]}"; do
    case "${s}" in
        tools) step_tools & TOOLS_PID=$! ;;   # network-bound; overlaps the SIF pulls
        *) "step_${s}" ;;
    esac
done
[ -n "${TOOLS_PID:-}" ] && wait "${TOOLS_PID}"
summary | tee "${HM_SHARED}/READY-${CLUSTER}.txt"
hm_log "fleet_stage done"
