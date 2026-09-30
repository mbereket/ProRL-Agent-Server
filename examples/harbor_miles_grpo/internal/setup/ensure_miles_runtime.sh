#!/usr/bin/env bash
# The Miles runtime: the pinned Apptainer SIF of radixark/miles, run through
# examples/miles_runtime/{mrun,ray_node.sh}. setup.sh is idempotent and
# flock-guarded (apptainer install, image pull, version manifest). Sourced by
# launch.sh; exports MILES_RUNTIME_DIR, MILES_STACK_ROOT, MILES_SIF, MILES_APPTAINER_BIN.
set -euo pipefail
SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=../../../harbor_slime_grpo/internal/setup/common.sh
source "${PROJECT_ROOT}/examples/harbor_slime_grpo/internal/setup/common.sh"

MILES_RUNTIME_DIR="${MILES_RUNTIME_DIR:-${PROJECT_ROOT}/examples/miles_runtime}"
[ -x "${MILES_RUNTIME_DIR}/mrun" ] || die "Miles runtime scripts not found in ${MILES_RUNTIME_DIR}"
log "miles runtime"
bash "${MILES_RUNTIME_DIR}/setup.sh" apptainer image manifest
# lib.sh resolves the stack root (CLUSTER_USER_ROOT/miles/stack) and the SIF path from pins.env.
eval "$(bash -c "source '${MILES_RUNTIME_DIR}/lib.sh' >/dev/null && echo MILES_STACK_ROOT=\${MILES_STACK_ROOT} MILES_SIF=\${MILES_SIF} MILES_APPTAINER_BIN=\$(mr_apptainer_bin)")"
emit_export MILES_RUNTIME_DIR "${MILES_RUNTIME_DIR}"
emit_export MILES_STACK_ROOT "${MILES_STACK_ROOT}"
emit_export MILES_SIF "${MILES_SIF}"
emit_export MILES_APPTAINER_BIN "${MILES_APPTAINER_BIN}"
[ -x "${MILES_APPTAINER_BIN}" ] || die "no apptainer at ${MILES_APPTAINER_BIN}"
info "image ${MILES_SIF}"
