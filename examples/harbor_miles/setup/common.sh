# shellcheck shell=bash
# Shared job environment for harbor_miles (sourced; CLUSTER must be set).
# Everything writable goes under HM_ROOT; HM_SHARED is only ever read.
HM_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
export HM_EXAMPLE_DIR
# shellcheck source=../cluster/clusters.sh
source "${HM_EXAMPLE_DIR}/cluster/clusters.sh"

hm_log() { echo "[harbor_miles $(date +%H:%M:%S) $(hostname -s)] $*" >&2; }
# Layered run configuration: hm_load_config <experiment.env> (recipe > layout > dataset > experiment, then derived knobs).
# shellcheck source=config.sh
source "${HM_EXAMPLE_DIR}/setup/config.sh"
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

# Port choice. dfw's ephemeral range is 9000-65000 (any outbound socket can hold a port in it), so fixed listen
# ports go above it: Ray 65010-65458 (miles_runtime/ray_node.sh), session servers 65460-65491, agent server 65500.
# hm_ports_free BASE COUNT: every port BASE..BASE+COUNT-1 can be bound right now (SO_REUSEADDR, as uvicorn binds).
hm_ports_free() {
    command -v python3 >/dev/null || return 0
    python3 -c 'import socket, sys
b, n = int(sys.argv[1]), int(sys.argv[2])
for p in range(b, b + n):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: s.bind(("0.0.0.0", p))
    except OSError: sys.exit(1)
    finally: s.close()' "$1" "$2"
}
# hm_free_port PREFERRED: PREFERRED if free, else a kernel-assigned free port.
hm_free_port() {
    if hm_ports_free "$1" 1; then echo "$1"; return; fi
    python3 -c 'import socket; s = socket.socket(); s.bind(("0.0.0.0", 0)); print(s.getsockname()[1]); s.close()'
}

# This node's cluster-reachable IPv4 address: what the hostname resolves to (same as the Ray
# head address), never a link-local 169.254.x interface (dfw lists one first in hostname -I).
hm_node_ip() {
    local ip
    ip="$(getent ahostsv4 "$(hostname)" 2>/dev/null | awk 'NR==1{print $1}')"
    [ -n "${ip}" ] || ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^169\.254\.' | grep -v '^127\.' | head -1)"
    echo "${ip}"
}
