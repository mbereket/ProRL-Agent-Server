#!/usr/bin/env bash
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
MILES_NO_BASE_PATCHES=1 "${MR}/mrun" --no-nv -- python3 - <<'PY'
import importlib.metadata as md
for d in sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower()):
    n = d.metadata["Name"]
    if n.lower().startswith(("opentelemetry", "prometheus")):
        print("DIST", n, d.version)
for n in ("ray", "sglang", "mlflow", "wandb", "opentelemetry-sdk", "opentelemetry-api"):
    try:
        req = [r for r in (md.requires(n) or []) if "opentelemetry" in r.lower()]
        print("REQ", n, req)
    except Exception as e:
        print("REQ", n, "ERR", e)
PY
