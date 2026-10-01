#!/usr/bin/env bash
# Prove a re-expressed config equivalent to the original it replaces: dry-render the ORIGINAL on its own launcher commit and the
# NEW one on this checkout, then compare (tools/config_equiv.py; exit 0 = only cosmetic differences).
#
#   tools/config_equiv.sh OLD_REF OLD_CONFIG NEW_CONFIG [--cluster C] [--nodes N] [--old-hm-root DIR] [--out DIR]
#   e.g. tools/config_equiv.sh origin/miles-diag configs/diag-de4-27b-overfit8-2n-r3.env \
#            configs/experiments/diag-de4-27b-overfit8-r3.env --nodes 2
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OREF="${1:?OLD_REF}"; OCFG="${2:?OLD_CONFIG}"; NCFG="${3:?NEW_CONFIG}"; shift 3
CL=dfw; NN=""; OHMR=""; OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --cluster) CL="$2"; shift 2 ;; --nodes) NN="$2"; shift 2 ;; --old-hm-root) OHMR="$2"; shift 2 ;; --out) OUT="$2"; shift 2 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
OUT="${OUT:-${TMPDIR:-/tmp}/hm-equiv/$(basename "${NCFG}" .env)}"
mkdir -p "${OUT}"
bash "${HERE}/dry_render.sh" --ref "${OREF}" --cluster "${CL}" ${NN:+--nodes "${NN}"} ${OHMR:+--hm-root "${OHMR}"} \
    --out "${OUT}/old" "${OCFG}" > "${OUT}/old.render.txt"
head -1 "${OUT}/old.render.txt"
bash "${HERE}/dry_render.sh" --cluster "${CL}" ${NN:+--nodes "${NN}"} --out "${OUT}/new" "${NCFG}" > "${OUT}/new.render.txt"
head -2 "${OUT}/new.render.txt"
python3 "${HERE}/config_equiv.py" "${OUT}/old" "${OUT}/new"
