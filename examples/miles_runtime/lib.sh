# shellcheck shell=bash
# Shared helpers for the Miles runtime (sourced by setup.sh, mrun, ray_node.sh).
#
# Layout under MILES_STACK_ROOT (one per cluster, on lustre):
#   apptainer/<ver>/bin/apptainer   unprivileged Apptainer (when none is on PATH)
#   images/<name>.sif               the pinned Miles image (+ .versions manifest)
#   overlays/<hash>/                patched source trees / extra pure-python deps (mrun --patches)
#   models/                         shared model weights (HF layout)
#   cache/                          apptainer pull cache + tmp
MR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./pins.env
source "${MR_DIR}/pins.env"

mr_log() { printf '[miles-runtime %s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
mr_die() { mr_log "FATAL: $*"; exit 1; }

# Resolve MILES_STACK_ROOT: explicit > CLUSTER_USER_ROOT (polar-slime cluster env) > known lustre roots.
mr_stack_root() {
    if [ -n "${MILES_STACK_ROOT:-}" ]; then echo "${MILES_STACK_ROOT}"; return; fi
    if [ -n "${CLUSTER_USER_ROOT:-}" ]; then echo "${CLUSTER_USER_ROOT}/miles/stack"; return; fi
    local base
    for base in /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/"${USER}" \
                /lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/"${USER}"; do
        if [ -d "${base}/miles/stack" ]; then echo "${base}/miles/stack"; return; fi
    done
    mr_die "cannot resolve MILES_STACK_ROOT; export it (e.g. <lustre user root>/miles/stack)"
}

MILES_STACK_ROOT="$(mr_stack_root)"
export MILES_STACK_ROOT
_mr_digest="${MILES_IMAGE_DIGEST#sha256:}"
MILES_SIF_NAME="miles-${MILES_IMAGE_TAG}-${_mr_digest:0:12}"
MILES_SIF="${MILES_SIF:-${MILES_STACK_ROOT}/images/${MILES_SIF_NAME}.sif}"
export MILES_SIF

# Apptainer binary: MILES_APPTAINER_BIN > PATH > unprivileged install under the stack root.
mr_apptainer_bin() {
    if [ -n "${MILES_APPTAINER_BIN:-}" ]; then echo "${MILES_APPTAINER_BIN}"; return; fi
    local b
    b="$(command -v apptainer 2>/dev/null || true)"
    if [ -n "${b}" ]; then echo "${b}"; return; fi
    echo "${MILES_STACK_ROOT}/apptainer/${APPTAINER_VERSION}/bin/apptainer"
}

mr_sha256() { sha256sum | awk '{print $1}'; }

# Kernel-driver major version on this node ("" when no GPU is visible).
mr_driver_major() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1; }

# Directory holding the forward-compat libcuda.so.1 when this node needs it (driver < 580 for the
# CUDA 13 image), else "". MILES_CUDA_COMPAT=0 disables, =1 forces.
MILES_CUDA_COMPAT_ROOT="${MILES_STACK_ROOT}/cuda-compat/${MILES_CUDA_COMPAT_VERSION}"
mr_cuda_compat_needed() {
    case "${MILES_CUDA_COMPAT:-auto}" in 0) return 1 ;; 1) return 0 ;; esac
    local maj; maj="$(mr_driver_major)"
    [ -n "${maj}" ] && [ "${maj}" -lt 580 ]
}
mr_cuda_compat_libdir() {
    mr_cuda_compat_needed || return 0
    local lib; lib="$(find "${MILES_CUDA_COMPAT_ROOT}" -name libcuda.so.1 2>/dev/null | head -1)"
    [ -n "${lib}" ] || mr_die "driver $(mr_driver_major) needs CUDA forward-compat libs; run setup.sh compat"
    dirname "${lib}"
}

# ---- persistent JIT caches (TileLang/Triton/Inductor/flashinfer/sglang/CUDA) across jobs --------------------
# mrun keeps all JIT caches under node-local MRUN cache_root (lustre caches shared live by concurrent jobs corrupt).
# mr_jit_seed copies the newest published snapshot in; mr_jit_publish tars the node's cache back (atomic rename,
# last writer wins; snapshots only ever grow when jobs run back to back). Keyed by image + GPU compute capability.
mr_cache_root() { echo "${MRUN_CACHE_ROOT:-/tmp/miles-${USER}-${SLURM_JOB_ID:-local}}"; }
mr_jit_dir() {
    local cc; cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)"
    echo "${MILES_STACK_ROOT}/jitcache/${MILES_SIF_NAME}-sm${cc:-x}"
}
mr_jit_seed() {
    [ "${MILES_JIT_CACHE:-1}" = 1 ] || return 0
    local d snap root; d="$(mr_jit_dir)"; snap="${d}/latest.tar"; root="$(mr_cache_root)"
    [ -s "${snap}" ] || { mr_log "jit cache: no snapshot at ${snap} (cold start)"; return 0; }
    mkdir -p "${root}" && tar -C "${root}" -xf "${snap}" 2>/dev/null \
        && mr_log "jit cache: seeded ${root} from ${snap} ($(du -sh "${snap}" | cut -f1))"
}
mr_jit_publish() {
    [ "${MILES_JIT_CACHE:-1}" = 1 ] || return 0
    local d root tmp; d="$(mr_jit_dir)"; root="$(mr_cache_root)"
    [ -d "${root}" ] || return 0
    mkdir -p "${d}"; tmp="${d}/.latest.tar.$$.$(hostname -s)"
    # skip sockets/locks; keep compiled artifacts only
    if tar -C "${root}" --exclude='*.lock' --exclude='*.sock' -cf "${tmp}" . 2>/dev/null; then
        mv -f "${tmp}" "${d}/latest.tar" && mr_log "jit cache: published $(du -sh "${d}/latest.tar" | cut -f1) -> ${d}/latest.tar"
    else
        rm -f "${tmp}"
    fi
}
