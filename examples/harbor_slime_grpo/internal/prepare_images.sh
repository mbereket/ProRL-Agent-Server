#!/usr/bin/env bash
# Pull every task image as an Apptainer SIF.
#
#   prepare_images.sh <images.txt> <sif_dir> [jobs]
#
# images.txt: "<docker_ref>\t<sif_name>" per line (prepare_tasks.py). Existing
# SIFs are kept. HARBOR_SIF_SEED_DIR (optional) is searched first for a SIF of
# the same image under the name "<ref with / and : replaced by _>.sif" (the
# layout Harbor's singularity environment caches into), which is linked instead
# of re-pulled. Registry auth: apptainer reads APPTAINER_DOCKER_USERNAME and
# APPTAINER_DOCKER_PASSWORD; set them for private images or Docker Hub rate limits.
# Run this where the registry is reachable (login node), not inside a job.
set -euo pipefail
MANIFEST="${1:?usage: prepare_images.sh <images.txt> <sif_dir> [jobs]}"
SIF_DIR="${2:?usage: prepare_images.sh <images.txt> <sif_dir> [jobs]}"
JOBS="${3:-${APPTAINER_PREPARE_JOBS:-4}}"
# SIF creation is mksquashfs compression of a 1-6 GB tree; single-threaded it
# dominates a pull. Give each build a share of the node's cores.
export APPTAINER_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS:-$(( $(nproc) / JOBS > 0 ? $(nproc) / JOBS : 1 ))}"
mkdir -p "${SIF_DIR}"
APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || command -v singularity || true)}"

pull_one() {
    local ref="$1" sif="$2" target="${SIF_DIR}/$2"
    if [ -s "${target}" ]; then echo "present: ${sif}"; return 0; fi
    if [ -n "${HARBOR_SIF_SEED_DIR:-}" ]; then
        local seed="${HARBOR_SIF_SEED_DIR}/$(echo "${ref}" | tr '/:' '__').sif"
        if [ -s "${seed}" ]; then echo "seeding ${sif} from ${seed}"; ln -sf "${seed}" "${target}"; return 0; fi
    fi
    [ -n "${APPTAINER_BIN}" ] && [ -x "${APPTAINER_BIN}" ] || { echo "ERROR: apptainer not found (needed to pull ${ref})" >&2; return 1; }
    local uri="${ref}"
    case "${ref}" in docker://*|oras://*|library://*|docker-daemon:*) ;; *) uri="docker://${ref}" ;; esac
    echo "pulling ${ref} -> ${sif}"
    # Pull to a temp name so a killed pull never leaves a truncated SIF behind.
    # OCI layers go through APPTAINER_CACHEDIR (set by the pipeline on the work
    # root), so layers shared between images are downloaded once.
    "${APPTAINER_BIN}" pull "${target}.part" "${uri}" >/dev/null && mv "${target}.part" "${target}"
}
export -f pull_one
export SIF_DIR APPTAINER_BIN HARBOR_SIF_SEED_DIR

# One pull per line, JOBS at a time.
grep -v '^\s*$' "${MANIFEST}" | tr '\t' ' ' | xargs -P "${JOBS}" -L 1 bash -c 'pull_one "$0" "$1"'

missing=0
while IFS=$'\t' read -r ref sif; do
    [ -n "${ref}" ] || continue
    [ -s "${SIF_DIR}/${sif}" ] || { echo "MISSING: ${sif} (${ref})" >&2; missing=$((missing + 1)); }
done < "${MANIFEST}"
[ "${missing}" -eq 0 ] || { echo "ERROR: ${missing} image(s) missing in ${SIF_DIR}" >&2; exit 1; }
echo "all images present in ${SIF_DIR}"
