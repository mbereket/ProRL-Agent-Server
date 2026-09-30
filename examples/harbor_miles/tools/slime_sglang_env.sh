# shellcheck shell=bash
# Validation only: borrow the polar-slime stack's SGLang (read-only) to serve a model
# for Harbor sandbox tests before the Miles runtime exists. Sourced after common.sh.
export VIRTUAL_ENV="${HM_SHARED}/ProRL-Agent-Server/.venv"
_cuda="${HM_SHARED}/cuda-toolkit-13.0/.pixi/envs/default"
export CUDA_HOME="${_cuda}"
export PATH="${VIRTUAL_ENV}/bin:${_cuda}/bin:${PATH}"
export LIBRARY_PATH="${_cuda}/targets/x86_64-linux/lib:${_cuda}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export CPATH="${_cuda}/targets/x86_64-linux/include:${_cuda}/include${CPATH:+:${CPATH}}"
export LD_LIBRARY_PATH="${_cuda}/lib:${_cuda}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [ "${HM_CUDA_COMPAT}" = 1 ]; then
    _compat="$(dirname "$(find "${HM_SHARED}"/cuda-compat-* -name 'libcuda.so.1' 2>/dev/null | head -1)")"
    [ -n "${_compat}" ] && export LD_LIBRARY_PATH="${_compat}:${LD_LIBRARY_PATH}"
fi
# JIT/compile caches: per job, never shared with other runs.
_jc="${HM_ROOT}/cache/jit-${SLURM_JOB_ID:-local}"
export TRITON_CACHE_DIR="${_jc}/triton" TORCHINDUCTOR_CACHE_DIR="${_jc}/inductor"
export FLASHINFER_WORKSPACE_BASE="${_jc}/flashinfer" TORCH_EXTENSIONS_DIR="${_jc}/torch_ext"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${FLASHINFER_WORKSPACE_BASE}" "${TORCH_EXTENSIONS_DIR}"
export QWEN35_9B_SNAPSHOT="${HM_SHARED}/hf_home/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
unset _cuda _compat _jc
