# shellcheck shell=bash
# Per-cluster paths for the harbor_miles example (sourced in jobs with CLUSTER set).
#
#   HM_ROOT         this example's writable work root on the cluster (venvs, runs, caches)
#   HM_SHARED       the polar-slime work root; used READ-ONLY here (SIFs, apptainer, uv,
#                   model snapshots, task packages). Nothing under it is ever written.
#   HM_SIF_DIRS     colon-separated read-only SIF directories searched before any pull
#   HM_APPTAINER    unprivileged apptainer install (bin/singularity + bin/apptainer)
#   HM_CUDA_COMPAT  1 when the driver needs the CUDA-13 forward-compat libraries
: "${CLUSTER:?CLUSTER must be set (hel|dfw)}"
case "${CLUSTER}" in
    hel)
        _user_root=/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/mbereket
        HM_CUDA_COMPAT=0
        ;;
    dfw)
        _user_root=/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket
        HM_CUDA_COMPAT=1
        ;;
    *) echo "clusters.sh: unknown cluster ${CLUSTER}" >&2; return 2 ;;
esac
export HM_USER_ROOT="${_user_root}"
export HM_ROOT="${HM_ROOT:-${_user_root}/miles/path-b}"
export HM_SHARED="${HM_SHARED:-${_user_root}/prorl-harbor}"
export HM_SIF_DIRS="${HM_SIF_DIRS:-${_user_root}/bio-synth/cache/apptainer/harbor:${HM_SHARED}/harbor_sif_images}"
export HM_APPTAINER="${HM_APPTAINER:-${HM_SHARED}/apptainer/1.5.3}"
export HM_UV="${HM_UV:-${HM_SHARED}/bin/uv}"
export HM_CUDA_COMPAT
unset _user_root
