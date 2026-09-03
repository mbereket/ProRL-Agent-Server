#!/usr/bin/env bash
# Let the SGLang engine base port be overridden with SLIME_ENGINE_BASE_PORT.
# Slime v0.3.0 hardcodes 15000 and assigns engine ports sequentially without
# checking availability; on clusters where another service holds a port in
# that range, one engine fails to bind and the run dies. Default stays 15000.
set -euo pipefail
SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)/slime}"
f="${SLIME_DIR}/slime/ray/rollout.py"
[ -f "${f}" ] || { echo "ERROR: ${f} not found" >&2; exit 1; }
if grep -q 'SLIME_ENGINE_BASE_PORT' "${f}"; then
    echo "slime engine base-port override already applied"; exit 0
fi
grep -q 'base_port = max(port_cursors.values()) if port_cursors else 15000' "${f}" \
    || { echo "ERROR: expected base_port line not found in ${f} (slime version changed?)" >&2; exit 1; }
sed -i 's|base_port = max(port_cursors.values()) if port_cursors else 15000|base_port = max(port_cursors.values()) if port_cursors else int(os.environ.get("SLIME_ENGINE_BASE_PORT", "15000"))|' "${f}"
grep -q '^import os$' "${f}" || sed -i '0,/^import /s//import os\nimport /' "${f}"
echo "applied slime engine base-port override (SLIME_ENGINE_BASE_PORT) to ${f}"
