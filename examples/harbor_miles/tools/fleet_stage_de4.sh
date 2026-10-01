#!/usr/bin/env bash
# FLEET: stage eval-v2 de4-v1-k1 NORMAL (97 tasks) + the bio-synth-harbor-codex:de-v2 SIF on a FLEET cluster (aws-iad first),
# with the same layout the staging worker uses on hel/dfw:
#   ${HM_SHARED}/tasks/de4-v1-k1-normal/{harbor/<97>, de4-v1-k1-normal-tasks.txt, SOURCE.txt, ...}
#   ${HM_SHARED}/harbor_sif_images/gitlab-master.nvidia.com_mbereket_bio-synth_bio-synth-harbor-codex_de-v2.sif
# Tasks come from ${HM_SHARED}/incoming/de4-v1-k1-normal.tgz (uploaded with scput). The SIF is pulled from the GitLab registry
# with GITLAB_TOKEN from the cluster-side ~/.secrets (loaded by common.sh into the environment; never on a command line or
# in a file), building in /dev/shm or node-local disk. Prints env presence (names only).
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
NAME=gitlab-master.nvidia.com_mbereket_bio-synth_bio-synth-harbor-codex_de-v2.sif
REF=gitlab-master.nvidia.com/mbereket/bio-synth/bio-synth-harbor-codex:de-v2
DT="${HM_SHARED}/tasks/de4-v1-k1-normal"; DS="${HM_SHARED}/harbor_sif_images"
mkdir -p "${HM_SHARED}/tasks" "${DS}"
if [ -f "${DT}/de4-v1-k1-normal-tasks.txt" ] && [ "$(ls "${DT}/harbor" 2>/dev/null | wc -l)" -ge 97 ]; then
    hm_log "de4 tasks: present ($(ls "${DT}/harbor" | wc -l))"
else
    tar -xzf "${HM_SHARED}/incoming/de4-v1-k1-normal.tgz" -C "${HM_SHARED}/tasks/" && hm_log "de4 tasks: $(ls "${DT}/harbor" | wc -l) task dirs, $(du -sh "${DT}" | cut -f1)"
fi
for v in NVINF_API_KEY HF_TOKEN GITLAB_TOKEN; do if [ -n "${!v:-}" ]; then hm_log "env ${v}: set"; else hm_log "env ${v}: UNSET"; fi; done
hm_log "gitlab-master reachability: $(curl -s -o /dev/null -m 15 -w '%{http_code}' https://gitlab-master.nvidia.com/v2/ || echo FAIL)"
if [ -s "${DS}/${NAME}" ]; then
    hm_log "de4 SIF: present $(ls -l "${DS}/${NAME}" | awk '{print $5}') bytes"
elif [ -n "${GITLAB_TOKEN:-}" ]; then
    B=/dev/shm/fleet-de4-${SLURM_JOB_ID:-x}; mkdir -p "${B}/tmp" "${B}/cache"
    t0=${SECONDS}
    if APPTAINER_TMPDIR="${B}/tmp" APPTAINER_CACHEDIR="${B}/cache" APPTAINER_DOCKER_USERNAME="${GITLAB_USER:-mbereket}" \
       APPTAINER_DOCKER_PASSWORD="${GITLAB_TOKEN}" "${HM_APPTAINER}/bin/apptainer" pull -F "${DS}/.${NAME}.part" "docker://${REF}" >"${B}/pull.log" 2>&1; then
        mv -f "${DS}/.${NAME}.part" "${DS}/${NAME}"; hm_log "de4 SIF: pulled in $((SECONDS - t0)) s ($(ls -l "${DS}/${NAME}" | awk '{print $5}') bytes)"
    else
        hm_log "de4 SIF: PULL FAILED: $(tail -3 "${B}/pull.log" | tr '\n' ' ' | cut -c1-400)"
    fi
    rm -rf "${B}"
else
    hm_log "de4 SIF: MISSING and no GITLAB_TOKEN in ~/.secrets -> needs an upload"
fi
sha256sum "${DS}/${NAME}" 2>/dev/null | cut -c1-16
hm_log "de4 stage done"
