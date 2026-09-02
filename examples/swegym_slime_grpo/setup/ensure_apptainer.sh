#!/usr/bin/env bash
# Resolve the container runtime Polar uses for task sandboxes.
#
# Order: POLAR_APPTAINER_BIN if set → apptainer/singularity on PATH → an
# unprivileged Apptainer install under WORKROOT (official
# tools/install-unprivileged.sh; no root, needs user namespaces, which
# preflight verified). Exports POLAR_APPTAINER_BIN for run.sh and image prep.
# Sourced by launch_e2e.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

APPTAINER_VERSION="${APPTAINER_VERSION:-1.5.3}"
APPTAINER_INSTALLER_URL="https://raw.githubusercontent.com/apptainer/apptainer/v${APPTAINER_VERSION}/tools/install-unprivileged.sh"
APPTAINER_INSTALLER_SHA256="${APPTAINER_INSTALLER_SHA256:-a097956eafa6ab3dd843ee429d0b774ed029753bfadf013b8046556228fac6f2}"

log "apptainer"
if [ "${NEED_APPTAINER:-0}" != 1 ]; then
    bin="${POLAR_APPTAINER_BIN:-$(command -v apptainer 2>/dev/null || command -v singularity)}"
    emit_export POLAR_APPTAINER_BIN "${bin}"
    info "using ${bin} ($("${bin}" --version))"
    return 0 2>/dev/null || exit 0
fi

root="${WORKROOT}/apptainer/${APPTAINER_VERSION}"
bin="${root}/bin/apptainer"
if [ ! -x "${bin}" ]; then
    info "installing unprivileged Apptainer ${APPTAINER_VERSION} into ${root}"
    # The installer unpacks RPMs and needs rpm2cpio + cpio. Shim rpm2cpio via
    # busybox when the host lacks it (common on minimal images).
    command -v cpio >/dev/null 2>&1 || die "cpio is required for the unprivileged Apptainer install"
    tools="$(mktemp -d "${WORKROOT}/.apptainer-tools.XXXXXX")"
    if ! command -v rpm2cpio >/dev/null 2>&1; then
        command -v busybox >/dev/null 2>&1 || die "neither rpm2cpio nor busybox found; cannot unpack Apptainer RPMs"
        cat > "${tools}/rpm2cpio" <<SH
#!/bin/sh
set -eu
if [ "\$#" -eq 0 ] || [ "\${1:-}" = - ]; then a=\$(mktemp); trap 'rm -f "\$a"' EXIT; cat > "\$a"; exec $(command -v busybox) rpm2cpio "\$a"
else exec $(command -v busybox) rpm2cpio "\$@"; fi
SH
        chmod +x "${tools}/rpm2cpio"
    fi
    fetch "${APPTAINER_INSTALLER_URL}" "${tools}/install-unprivileged.sh"
    actual="$(sha256sum "${tools}/install-unprivileged.sh" | awk '{print $1}')"
    [ "${actual}" = "${APPTAINER_INSTALLER_SHA256}" ] || die "Apptainer installer checksum mismatch (${actual})"
    staged="${root}.staging"; rm -rf "${staged}"; mkdir -p "$(dirname "${root}")"
    PATH="${tools}:${PATH}" bash "${tools}/install-unprivileged.sh" -e -v "${APPTAINER_VERSION}" "${staged}"
    rm -rf "${root}"; mv "${staged}" "${root}"; rm -rf "${tools}"
fi
[ -x "${bin}" ] || die "Apptainer install did not produce ${bin}"
emit_export POLAR_APPTAINER_BIN "${bin}"
info "using ${bin} ($("${bin}" --version))"
