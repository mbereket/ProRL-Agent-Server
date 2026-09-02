#!/usr/bin/env bash
# Submit a multi-node run with plain sbatch.
#
#   bash multinode/sbatch_launch.sh --nodes 2 --partition batch --account <acct> [--time 04:00:00] [-- <extra sbatch args>]
#
# Every environment variable you export before calling this (WORKROOT, ports,
# MAX_TOKENS_PER_GPU, HF_CHECKPOINT, WANDB_API_KEY, ...) reaches the job via
# sbatch --export=ALL. The job runs multinode/head_entry.sh on the first node.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
NODES=2; PARTITION=""; ACCOUNT=""; TIME="04:00:00"; GPUS_PER_NODE=8; CPUS=""; MEM=""; JOB_NAME="swegym-grpo"; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --nodes) NODES="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --time) TIME="$2"; shift 2 ;;
        --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --cpus-per-task) CPUS="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
WORKROOT="${WORKROOT:-${PROJECT_ROOT}/tmp}"
LOG_DIR="${SLURM_LOG_DIR:-${WORKROOT}/joblogs}"
mkdir -p "${LOG_DIR}"
export WORKROOT
args=(--job-name "${JOB_NAME}" --nodes "${NODES}" --ntasks-per-node 1 --gpus-per-node "${GPUS_PER_NODE}"
      --time "${TIME}" --output "${LOG_DIR}/%j-%x.log" --export ALL)
[ -n "${PARTITION}" ] && args+=(--partition "${PARTITION}")
[ -n "${ACCOUNT}" ]   && args+=(--account "${ACCOUNT}")
[ -n "${CPUS}" ]      && args+=(--cpus-per-task "${CPUS}")
[ -n "${MEM}" ]       && args+=(--mem "${MEM}")
echo "sbatch ${args[*]} ${EXTRA[*]:-} -- head_entry.sh (logs: ${LOG_DIR})"
sbatch "${args[@]}" "${EXTRA[@]}" --wrap "bash ${SCRIPT_DIR}/head_entry.sh"
