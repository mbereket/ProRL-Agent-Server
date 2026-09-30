#!/usr/bin/env bash
# 2+ node check (one srun task per node): NCCL all-reduce across nodes (IB?) + ray_node.sh bring-up.
# Full logs: $MILES_STACK_ROOT/diag/multinode-<jobid>/
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
OUT="${MILES_STACK_ROOT}/diag/multinode-${SLURM_JOB_ID}"; mkdir -p "${OUT}"
nodes=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
head_ip="$(getent ahostsv4 "${nodes[0]}" | awk 'NR==1{print $1}')"
G="$(nvidia-smi --list-gpus | wc -l)"
log="${OUT}/nccl-node${SLURM_NODEID}.log"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET timeout 600 "${MR}/mrun" -- torchrun --nnodes "${#nodes[@]}" --nproc-per-node "${G}" \
    --node-rank "${SLURM_NODEID}" --master-addr "${head_ip}" --master-port "$((30000 + SLURM_JOB_ID % 20000))" \
    "${MR}/bench/nccl_test.py" > "${log}" 2>&1
echo "node ${SLURM_NODEID}: torchrun rc=$?"
grep -E "allreduce|NET/IB|NET/Socket|Using network|via NET" "${log}" | sort | uniq -c | sort -rn | head -12
sleep 5
"${MR}/ray_node.sh" -- python3 -c "import ray; ray.init(); print('RAY_OK nodes', len([n for n in ray.nodes() if n['Alive']]), 'GPU', ray.cluster_resources().get('GPU'))"
echo "multinode_check done rc=$?"
