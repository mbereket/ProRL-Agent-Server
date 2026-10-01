#!/usr/bin/env bash
# FLEET probe: host facts + Slurm limits + Miles runtime setup on a new cluster (draco/ord).
#   probe.sh [facts] [setup] [model]   (default: all three)
set -uo pipefail
MR="${SCOMPOSE_PKGS:?}/miles_runtime"
: "${MILES_STACK_ROOT:?}" "${MILES_OWNER_ROOT:?}"
export PYTHONUNBUFFERED=1
log() { printf '\n===== [fleet %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

facts() {
    log "host"
    hostname; uname -r; nproc; free -g | head -2
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
    nvidia-smi | head -4 | tail -2
    nvidia-smi topo -m 2>/dev/null | head -12
    log "disk"
    df -h /tmp /dev/shm "${MILES_STACK_ROOT%/miles/stack}" 2>/dev/null
    ls -d /raid /scratch /local* /mnt/* 2>/dev/null | head; df -h /raid /scratch 2>/dev/null
    log "apptainer / userns"
    command -v apptainer singularity enroot 2>/dev/null; apptainer --version 2>/dev/null
    cat /proc/sys/user/max_user_namespaces 2>/dev/null
    unshare -Ur true 2>&1 && echo "userns: unshare -Ur OK" || echo "userns: unshare -Ur FAILED"
    grep "^${USER}:" /etc/subuid /etc/subgid 2>/dev/null || echo "no subuid/subgid entry"
    command -v cpio rpm2cpio busybox 2>/dev/null
    log "network egress"
    for u in https://pypi.org/simple/ https://huggingface.co https://registry-1.docker.io/v2/ https://developer.download.nvidia.com https://raw.githubusercontent.com; do
        printf '%-45s %s\n' "$u" "$(curl -s -o /dev/null -m 15 -w '%{http_code} %{time_total}s' "$u" || echo FAIL)"
    done
    log "interconnect"
    ls /dev/infiniband 2>/dev/null | head -5 || echo "no /dev/infiniband"
    command -v ibv_devinfo >/dev/null && ibv_devinfo -l 2>/dev/null | head -12
    ls /sys/class/infiniband 2>/dev/null | head; ip -br link 2>/dev/null | head -12
    cat /proc/sys/net/ipv4/ip_local_port_range
    log "slurm limits"
    scontrol show partition interactive interactive_singlenode batch_singlenode backfill_singlenode backfill_block1 2>/dev/null \
        | grep -E 'PartitionName|MaxTime|MaxNodes|QoS|DefaultTime|AllowQos|TRES='
    sacctmgr -nP show assoc user="${USER}" format=cluster,account,partition,qos,maxjobs,maxsubmit,grptres,maxtres 2>/dev/null | head -20
    sacctmgr -nP show qos format=name,maxjobspu,maxsubmitpu,maxtrespu,maxwall,priority 2>/dev/null | head -40
    squeue -u "${USER}" -h -o '%i %P %j %T %D' 2>/dev/null
}

setup() {
    log "miles runtime setup (apptainer compat image manifest smoke pidns nested)"
    bash "${MR}/setup.sh" apptainer compat image manifest smoke pidns nested
}

model() {
    log "stage Qwen3.5-9B"
    mkdir -p "${MILES_STACK_ROOT}/models"
    "${MR}/mrun" -- bash -c '
        export HF_HOME='"${MILES_STACK_ROOT}"'/hf_home HF_HUB_ENABLE_HF_TRANSFER=0
        d='"${MILES_STACK_ROOT}"'/models/Qwen3.5-9B
        if [ -f "$d/config.json" ]; then echo "present: $d"; else
            t0=$(date +%s)
            python3 -c "from huggingface_hub import snapshot_download as s; s(\"Qwen/Qwen3.5-9B\", local_dir=\"$d.partial.$$\")" && mv "$d.partial.$$" "$d"
            echo "downloaded in $(( $(date +%s) - t0 )) s"
        fi
        du -sh "$d"'
}

steps=("$@"); [ "${#steps[@]}" -eq 0 ] && steps=(facts setup model)
for s in "${steps[@]}"; do "$s"; done
log "done"
