#!/usr/bin/env bash
# run_many.sh SUITE PATCH_DIR ARMS_FILE [ARMS_FILE...] : run several arms files back to back in one allocation.
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
suite="$1"; pd="$2"; shift 2
for f in "$@"; do bash "${MR}/bench/run.sh" "${suite}" "${f}" "${pd}"; done
