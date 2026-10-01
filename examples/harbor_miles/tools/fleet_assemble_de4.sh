#!/usr/bin/env bash
# FLEET: reassemble the de4 SIF uploaded in 80 MB chunks (parallel scput; one SFTP stream is ~170 KB/s to aws-iad) into
# ${HM_SHARED}/harbor_sif_images, verifying every chunk and the whole file against the source sha256
# (f93335ae609f89e2..., the hel/draco copy). Runs fleet_stage.sh sifs first (useful work while the upload finishes), then
# waits up to WAIT_MIN for the chunks.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
IN="${HM_SHARED}/incoming/de4sif"; DS="${HM_SHARED}/harbor_sif_images"
NAME=gitlab-master.nvidia.com_mbereket_bio-synth_bio-synth-harbor-codex_de-v2.sif
STAGE_JOBS="${STAGE_JOBS:-24}" bash "${HERE}/fleet_stage.sh" sifs
if [ -s "${DS}/${NAME}" ]; then hm_log "de4 SIF: present"; else
    end=$(( $(date +%s) + ${WAIT_MIN:-90} * 60 ))
    until [ -s "${IN}/FULL.sha256" ] && [ -s "${IN}/SHA256SUMS" ] && (cd "${IN}" && sha256sum -c --quiet SHA256SUMS >/dev/null 2>&1); do
        [ "$(date +%s)" -gt "${end}" ] && { hm_log "de4 SIF: chunks incomplete after ${WAIT_MIN:-90} min: $(ls "${IN}" 2>/dev/null | wc -l) files"; break; }
        sleep 60
    done
    if (cd "${IN}" && sha256sum -c --quiet SHA256SUMS); then
        cat "${IN}"/de4sif.part.* > "${DS}/.${NAME}.part.${SLURM_JOB_ID:-x}"
        got="$(sha256sum "${DS}/.${NAME}.part.${SLURM_JOB_ID:-x}" | awk '{print $1}')"
        if [ "${got}" = "$(cat "${IN}/FULL.sha256")" ]; then
            mv -f "${DS}/.${NAME}.part.${SLURM_JOB_ID:-x}" "${DS}/${NAME}"; hm_log "de4 SIF: assembled + verified ${got:0:16} ($(stat -c %s "${DS}/${NAME}") bytes)"
        else hm_log "de4 SIF: CHECKSUM MISMATCH ${got:0:16}"; fi
    fi
fi
hm_log "assemble_de4 done"
