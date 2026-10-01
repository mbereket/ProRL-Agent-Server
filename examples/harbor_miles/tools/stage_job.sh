#!/usr/bin/env bash
# CPU-only SIF staging job (resumable: present SIFs are skipped).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${HERE}/../setup/common.sh"
bash "${HERE}/stage_sifs.sh" "${HM_EXAMPLE_DIR}/${REFS_FILE:-configs/swegym-dfw-staged93-refs.txt}" "${STAGE_JOBS:-8}"
