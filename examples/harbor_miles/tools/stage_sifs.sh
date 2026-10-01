#!/usr/bin/env bash
# Pull docker images as SIFs into ${HM_ROOT}/sif_cache with Harbor's own cache names
# ("/" and ":" -> "_", ":latest" added), i.e. exactly what Harbor's singularity backend would
# create on first use — done ahead of time so trials never wait on a pull.
#   stage_sifs.sh <refs-file> [jobs]
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
REFS="${1:?refs file}"; JOBS="${2:-4}"
OUT="${HM_ROOT}/sif_cache"; mkdir -p "${OUT}"
export APPTAINER_MKSQUASHFS_PROCS="${APPTAINER_MKSQUASHFS_PROCS:-8}"
pull_one() {
    local ref="$1" name
    case "${ref##*/}" in *:*) ;; *) ref="${ref}:latest" ;; esac
    name="$(echo "${ref}" | tr '/:' '__').sif"
    [ -s "${OUT}/${name}" ] && { echo "present ${name}"; return 0; }
    local t0=${SECONDS}
    if "${HM_APPTAINER}/bin/apptainer" pull -F "${OUT}/${name}.part.$$" "docker://${ref}" >/dev/null 2>"${OUT}/${name}.err"; then
        mv "${OUT}/${name}.part.$$" "${OUT}/${name}"; echo "pulled ${name} in $((SECONDS - t0)) s"
    else
        echo "FAILED ${ref}: $(tail -2 "${OUT}/${name}.err" | tr '\n' ' ')"
    fi
}
export -f pull_one; export OUT HM_APPTAINER
grep -v '^\s*$' "${REFS}" | xargs -P "${JOBS}" -I{} bash -c 'pull_one "$@"' _ {}
echo "staged: $(ls "${OUT}"/*.sif 2>/dev/null | wc -l) SIFs in ${OUT}"
