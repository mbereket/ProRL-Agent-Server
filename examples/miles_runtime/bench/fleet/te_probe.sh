#!/usr/bin/env bash
# A100 attention-backend probe inside the Miles runtime.
set -uo pipefail
MRUN="${SCOMPOSE_PKGS}/miles_runtime/mrun"; HERE="$(cd "$(dirname "$0")" && pwd)"
"${MRUN}" -- python3 "${HERE}/te_attn_test.py"
