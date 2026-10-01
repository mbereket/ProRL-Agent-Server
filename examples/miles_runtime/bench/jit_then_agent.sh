#!/usr/bin/env bash
# One allocation: JIT-cache warm check (new job, seeded snapshot) then the agentic rollout-capacity sweep.
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
bash "${MR}/bench/run.sh" q35-9b-real arms-jit-warm-hel.txt bench/patches
bash "${MR}/bench/agent_suite.sh" agent-cap-v1
