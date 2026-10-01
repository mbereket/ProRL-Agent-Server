# shellcheck shell=bash
# Shared job environment for harbor_miles (sourced; CLUSTER must be set).
# Everything writable goes under HM_ROOT; HM_SHARED is only ever read.
HM_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
export HM_EXAMPLE_DIR
# shellcheck source=../cluster/clusters.sh
source "${HM_EXAMPLE_DIR}/cluster/clusters.sh"

hm_log() { echo "[harbor_miles $(date +%H:%M:%S) $(hostname -s)] $*" >&2; }
hm_die() { hm_log "FATAL: $*"; exit 1; }

export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
mkdir -p "${HM_ROOT}"/{cache,runs,uv_cache,uv_python,apptainer_config}
# Node-local temp: short paths (ZMQ IPC sockets are limited to 107 chars; the dfw lustre root alone
# is ~80) and fast for Harbor's per-trial staging dirs (bind-mounted into sandboxes on this node).
export TMPDIR="/tmp/hm-${USER}-${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}"
export UV_CACHE_DIR="${HM_ROOT}/uv_cache"
export UV_PYTHON_INSTALL_DIR="${HM_ROOT}/uv_python"
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="${HM_ROOT}/cache"
export APPTAINER_CONFIGDIR="${HM_ROOT}/apptainer_config"
export APPTAINER_CACHEDIR="${HM_ROOT}/cache/apptainer"
# Node-local scratch for apptainer's own temp files (not lustre).
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/hm-apptainer-${USER}-${SLURM_JOB_ID:-local}}"
mkdir -p "${APPTAINER_TMPDIR}"
export PATH="${HM_APPTAINER}/bin:$(dirname "${HM_UV}"):${PATH}"
export HF_HOME="${HM_ROOT}/hf_home"

# Secrets (WANDB_API_KEY, NVINF_API_KEY, ...) come from the cluster-side ~/.secrets,
# never from the command line.
if [ -r "${HOME}/.secrets" ]; then
    set -a; source "${HOME}/.secrets"; set +a
fi

# Flock-guarded, stamp-checked one-time step: hm_once <name> <cmd...>
hm_once() {
    local name="$1"; shift
    local stamp="${HM_ROOT}/.stamps/${name}"
    mkdir -p "${HM_ROOT}/.stamps"
    [ -f "${stamp}" ] && return 0
    (
        flock -x 9
        [ -f "${stamp}" ] && exit 0
        "$@" && touch "${stamp}"
    ) 9>"${stamp}.lock"
}

# This node's cluster-reachable IPv4 address.
hm_node_ip() {
    local ip
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [ -n "${ip}" ] || ip="$(hostname -i | awk '{print $1}')"
    echo "${ip}"
}
