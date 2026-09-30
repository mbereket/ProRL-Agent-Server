#!/usr/bin/env bash
# Idempotent setup of the Miles runtime on one cluster (run inside a Slurm job on
# a compute node; the login nodes have no apptainer and weak egress).
#
#   setup.sh [all|apptainer|compat|image|manifest|smoke|pidns|nested]...   (default: all)
#
#   apptainer  unprivileged Apptainer under $MILES_STACK_ROOT/apptainer (skipped if one is on PATH)
#   compat     CUDA forward-compat libcuda when the kernel driver is < 580 (dfw, aws-iad)
#   image      docker://radixark/miles@<digest> -> $MILES_SIF (atomic, flock-guarded)
#   manifest   $MILES_SIF.versions: commits/versions baked into the image
#   smoke      GPU + imports + NCCL/IB visibility inside the SIF
#   nested     can a process inside the SIF start Apptainer sandboxes (exec + instance)?
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/lib.sh"

APPTAINER="$(mr_apptainer_bin)"
mkdir -p "${MILES_STACK_ROOT}"/{images,overlays,models,cache/apptainer,cache/tmp,joblogs}
export APPTAINER_CACHEDIR="${MILES_STACK_ROOT}/cache/apptainer"

step_apptainer() {
    if [ -x "${APPTAINER}" ]; then mr_log "apptainer: ${APPTAINER} ($("${APPTAINER}" --version))"; return; fi
    local root="${MILES_STACK_ROOT}/apptainer/${APPTAINER_VERSION}"
    [ "${APPTAINER}" = "${root}/bin/apptainer" ] || mr_die "MILES_APPTAINER_BIN=${APPTAINER} is not executable"
    command -v cpio >/dev/null || mr_die "cpio is required for the unprivileged Apptainer install"
    (
        flock -w 1800 9 || mr_die "timed out waiting for the apptainer install lock"
        [ -x "${root}/bin/apptainer" ] && exit 0
        local tools; tools="$(mktemp -d "${MILES_STACK_ROOT}/cache/tmp/apptainer-tools.XXXXXX")"
        if ! command -v rpm2cpio >/dev/null; then
            command -v busybox >/dev/null || mr_die "neither rpm2cpio nor busybox found"
            printf '#!/bin/sh\nset -eu\nif [ "$#" -eq 0 ] || [ "${1:-}" = - ]; then a=$(mktemp); trap '"'"'rm -f "$a"'"'"' EXIT; cat > "$a"; exec %s rpm2cpio "$a"\nelse exec %s rpm2cpio "$@"; fi\n' \
                "$(command -v busybox)" "$(command -v busybox)" > "${tools}/rpm2cpio"
            chmod +x "${tools}/rpm2cpio"
        fi
        curl -fsSL --retry 3 -o "${tools}/install-unprivileged.sh" \
            "https://raw.githubusercontent.com/apptainer/apptainer/v${APPTAINER_VERSION}/tools/install-unprivileged.sh"
        [ "$(sha256sum "${tools}/install-unprivileged.sh" | awk '{print $1}')" = "${APPTAINER_INSTALLER_SHA256}" ] \
            || mr_die "Apptainer installer checksum mismatch"
        rm -rf "${root}.staging"; mkdir -p "$(dirname "${root}")"
        PATH="${tools}:${PATH}" bash "${tools}/install-unprivileged.sh" -e -v "${APPTAINER_VERSION}" "${root}.staging"
        rm -rf "${root}"; mv "${root}.staging" "${root}"; rm -rf "${tools}"
    ) 9>"${MILES_STACK_ROOT}/cache/apptainer-install.lock"
    [ -x "${APPTAINER}" ] || mr_die "Apptainer install did not produce ${APPTAINER}"
    mr_log "apptainer: installed ${APPTAINER} ($("${APPTAINER}" --version))"
}

step_compat() {
    if ! mr_cuda_compat_needed; then mr_log "compat: driver $(mr_driver_major) runs CUDA 13 natively"; return; fi
    local root="${MILES_CUDA_COMPAT_ROOT}"
    if [ -z "$(find "${root}" -name libcuda.so.1 2>/dev/null)" ]; then
        (
            flock -w 1800 9 || mr_die "timed out waiting for the cuda-compat lock"
            [ -n "$(find "${root}" -name libcuda.so.1 2>/dev/null)" ] && exit 0
            local url="https://developer.download.nvidia.com/compute/cuda/redist/cuda_compat/linux-x86_64/cuda_compat-linux-x86_64-${MILES_CUDA_COMPAT_VERSION}-archive.tar.xz"
            mkdir -p "${root}.staging"
            curl -fsSL --retry 3 "${url}" | tar -xJ -C "${root}.staging" --strip-components=1
            mv "${root}.staging" "${root}"
        ) 9>"${MILES_STACK_ROOT}/cache/cuda-compat.lock"
    fi
    mr_log "compat: driver $(mr_driver_major) -> forward-compat libcuda from $(mr_cuda_compat_libdir)"
}

step_image() {
    if [ -s "${MILES_SIF}" ]; then mr_log "image: ${MILES_SIF} present ($(du -h "${MILES_SIF}" | cut -f1))"; return; fi
    (
        flock -w 10800 9 || mr_die "timed out waiting for the image pull lock"
        [ -s "${MILES_SIF}" ] && exit 0
        # Unpacking a ~24 GB (compressed) image needs ~3x that in scratch: node-local /tmp when it is big enough.
        local free_gb; free_gb="$(df -BG --output=avail /tmp | tail -1 | tr -dc 0-9)"
        if [ "${free_gb:-0}" -ge 200 ]; then
            APPTAINER_TMPDIR="$(mktemp -d /tmp/miles-sif-build.XXXXXX)"
        else
            APPTAINER_TMPDIR="$(mktemp -d "${MILES_STACK_ROOT}/cache/tmp/sif-build.XXXXXX")"
        fi
        export APPTAINER_TMPDIR
        mr_log "image: pulling ${MILES_IMAGE_REPO}@${MILES_IMAGE_DIGEST} (tag ${MILES_IMAGE_TAG}); tmp ${APPTAINER_TMPDIR} (/tmp free ${free_gb}G)"
        local t0=${SECONDS} part="${MILES_SIF}.partial.$$"
        rm -f "${part}"
        "${APPTAINER}" pull --force "${part}" "docker://${MILES_IMAGE_REPO}@${MILES_IMAGE_DIGEST}"
        mv -f "${part}" "${MILES_SIF}"
        rm -rf "${APPTAINER_TMPDIR}"
        mr_log "image: done in $((SECONDS - t0)) s -> ${MILES_SIF} ($(du -h "${MILES_SIF}" | cut -f1))"
    ) 9>"${MILES_SIF}.lock"
}

step_manifest() {
    local out="${MILES_SIF%.sif}.versions"
    [ -s "${out}" ] && { mr_log "manifest: ${out}"; cat "${out}"; return; }
    "${APPTAINER}" exec "${MILES_SIF}" bash -c '
        echo "image=${MILES_IMAGE_REPO:-}"; python3 --version
        for r in /root/miles /sgl-workspace/sglang /root/Megatron-LM; do
            echo "$r $(git -C $r rev-parse HEAD 2>/dev/null) $(git -C $r log -1 --format=%cd --date=short 2>/dev/null) $(git -C $r rev-parse --abbrev-ref HEAD 2>/dev/null)"
        done
        pip list 2>/dev/null | grep -iE "^(torch|sglang|sglang-router|sgl-kernel|transformer.engine|megatron|flash.attn|flashinfer|mbridge|ray|transformers|peft|triton|nvidia-nccl|apex|flash-linear|deep.gemm) " || true
        python3 -c "import torch;print(\"torch.cuda\", torch.version.cuda, \"nccl\", torch.cuda.nccl.version())"
    ' > "${out}.tmp" 2>&1 && mv "${out}.tmp" "${out}"
    mr_log "manifest: ${out}"; cat "${out}"
}

step_smoke() {
    mr_log "smoke: host $(hostname), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1), /tmp free $(df -h --output=avail /tmp | tail -1)"
    "${MR_DIR}/mrun" -- bash -c '
        set -e
        nvidia-smi -L | head -2
        python3 - <<PY
import time, torch
t = time.time()
import sglang, miles, megatron.core, transformer_engine, megatron.bridge  # noqa: F401
print("imports ok in %.1fs; torch %s cuda %s devices %d" % (time.time() - t, torch.__version__, torch.version.cuda, torch.cuda.device_count()))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize()
t = time.time(); [x @ x for _ in range(50)]; torch.cuda.synchronize(); dt = time.time() - t
print("bf16 matmul %.0f TFLOP/s" % (50 * 2 * 4096**3 / dt / 1e12))
PY
        ls /dev/infiniband 2>/dev/null | head -3 || echo "no /dev/infiniband"
        (command -v ibv_devinfo >/dev/null && ibv_devinfo -l) || ls /usr/lib/x86_64-linux-gnu/libibverbs* /usr/lib/x86_64-linux-gnu/libibverbs 2>/dev/null | head -5 || echo "no libibverbs"
        env | grep -E "^(PATH|LD_LIBRARY_PATH|PYTHONPATH|HOME|CUDA_HOME)=" | sed "s/^/env: /"
    '
}

step_pidns() {
    mr_log "pidns: apptainer --pid (used by ray_node.sh for clean teardown)"
    MRUN_APPTAINER_ARGS=--pid "${MR_DIR}/mrun" -- bash -c 'echo "pid-in-ns=$$"; sleep 300 & echo "child $!"; nvidia-smi -L | head -1' \
        && mr_log "pidns: ok" || mr_log "pidns: FAILED"
}

step_nested() {
    local probe="${MILES_STACK_ROOT}/cache/busybox.sif"
    [ -s "${probe}" ] || "${APPTAINER}" pull --force "${probe}" docker://busybox:1.36
    mr_log "nested: apptainer exec from inside the Miles SIF"
    "${MR_DIR}/mrun" -- bash -c "
        set -x
        '${APPTAINER}' --version
        '${APPTAINER}' exec '${probe}' sh -c 'echo nested-exec-ok \$(id -u)'
        '${APPTAINER}' instance start '${probe}' mr-nested-\$\$ && '${APPTAINER}' exec instance://mr-nested-\$\$ echo nested-instance-ok; '${APPTAINER}' instance stop mr-nested-\$\$ || true
    " || mr_log "nested: FAILED (see above)"
}

steps=("$@"); [ "${#steps[@]}" -eq 0 ] && steps=(all)
for s in "${steps[@]}"; do
    case "${s}" in
        all) step_apptainer; step_compat; step_image; step_manifest; step_smoke; step_pidns; step_nested ;;
        apptainer|compat|image|manifest|smoke|pidns|nested) "step_${s}" ;;
        *) mr_die "unknown step ${s}" ;;
    esac
done
