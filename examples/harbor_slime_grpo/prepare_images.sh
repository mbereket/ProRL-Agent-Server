#!/usr/bin/env bash
# Pull task images as Apptainer SIFs into the shared image directory.
#
#   bash prepare_images.sh <task_dir>  [sif_dir]     every image the task directory references
#   bash prepare_images.sh <run.yaml>  [sif_dir]     only the images of the tasks that run config selects
#
# sif_dir default: ${APPTAINER_IMAGE_DIR:-${WORKROOT}/harbor_sif_images}. SIF names
# follow Harbor's singularity cache ("/" and ":" -> "_"), so a Harbor cache
# directory works as sif_dir directly. Existing SIFs are kept. Run it where the
# registry is reachable; APPTAINER_JOBS pulls at a time (default 4). Registry
# auth: APPTAINER_DOCKER_USERNAME / APPTAINER_DOCKER_PASSWORD.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PROJECT_ROOT="$(cd -- "${HERE}/../.." && pwd)"
TARGET="${1:?usage: prepare_images.sh <task_dir|run.yaml> [sif_dir]}"
# shellcheck source=./internal/setup/common.sh
source "${HERE}/internal/setup/common.sh"

select=()
if [ -f "${TARGET}" ]; then   # run config: its task dir and subset
    export WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
    eval "$(config_python "${HERE}/internal/render.py" env "${TARGET}")"
    select=(--seed "${TASKS_SEED}")
    [ -n "${TASKS_N}" ] && select+=(--n "${TASKS_N}")
    [ -n "${TASK_IDS_FILE}" ] && select+=(--task-ids-file "${TASK_IDS_FILE}")
    [ -n "${EXCLUDE_IDS_FILE}" ] && select+=(--exclude-ids-file "${EXCLUDE_IDS_FILE}")
else
    TASKS_DIR="${TARGET}"
fi
SIF_DIR="${2:-${APPTAINER_IMAGE_DIR:-${WORKROOT:?set WORKROOT or pass sif_dir}/harbor_sif_images}}"
JOBS="${APPTAINER_JOBS:-4}"
# POLAR_APPTAINER_BIN, else PATH, else the unprivileged install setup puts under WORKROOT.
APPTAINER="${POLAR_APPTAINER_BIN:-$(command -v apptainer || command -v singularity || ls "${WORKROOT:-/nonexistent}"/apptainer/*/bin/apptainer 2>/dev/null | tail -1 || true)}"
[ -x "${APPTAINER}" ] || die "apptainer/singularity not found (set POLAR_APPTAINER_BIN, or run launch.sh <cfg> --setup-only which installs and pulls)"

mkdir -p "${SIF_DIR}"
LIST="$(mktemp)"; trap 'rm -f "${LIST}"' EXIT
config_python "${HERE}/internal/prepare_tasks.py" --tasks-dir "${TASKS_DIR}" --list-images ${select[@]+"${select[@]}"} > "${LIST}"
echo "$(wc -l < "${LIST}" | tr -d ' ') distinct image(s) for ${TASKS_DIR} -> ${SIF_DIR}"

# SIF creation is mksquashfs of a 1-6 GB tree; single-threaded it dominates a pull.
NCPU="$(nproc 2>/dev/null || echo 8)"
export APPTAINER_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS:-$(( NCPU / JOBS > 0 ? NCPU / JOBS : 1 ))}"
pull_one() {   # pull_one <docker_ref> <sif>
    local target="${SIF_DIR}/$2"
    if [ -s "${target}" ]; then echo "present: $2"; return 0; fi
    echo "pulling $1 -> $2"
    # Temp name so a killed pull never leaves a truncated SIF behind.
    "${APPTAINER}" pull "${target}.part" "docker://$1" >/dev/null && mv "${target}.part" "${target}"
}
export -f pull_one; export SIF_DIR APPTAINER
tr '\t' ' ' < "${LIST}" | xargs -P "${JOBS}" -L 1 bash -c 'pull_one "$0" "$1"'

missing=0
while IFS=$'\t' read -r ref sif; do
    [ -s "${SIF_DIR}/${sif}" ] || { echo "MISSING: ${sif} (${ref})" >&2; missing=$((missing + 1)); }
done < "${LIST}"
[ "${missing}" -eq 0 ] || die "${missing} image(s) missing in ${SIF_DIR}"
echo "all images present in ${SIF_DIR}"
