#!/usr/bin/env bash
# 2+ node check (one srun task per node): NCCL all-reduce across nodes (IB?) + ray_node.sh bring-up.
set -uo pipefail
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
source "${MR}/lib.sh"
nodes=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
head_ip="$(getent ahostsv4 "${nodes[0]}" | awk 'NR==1{print $1}')"
G="$(nvidia-smi --list-gpus | wc -l)"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET "${MR}/mrun" -- torchrun --nnodes "${#nodes[@]}" --nproc-per-node "${G}" \
    --node-rank "${SLURM_NODEID}" --master-addr "${head_ip}" --master-port 29555 "${MR}/bench/nccl_test.py" 2>&1 \
    | grep -E "allreduce|NET/|Using network|NCCL INFO Channel 00/0|comm .* rank 0 nRanks|error|Error" | grep -v "^\s*$" | head -40
sleep 5
"${MR}/ray_node.sh" -- python3 -c "import ray; ray.init(); r = ray.cluster_resources(); print('RAY_OK nodes', len([n for n in ray.nodes() if n['Alive']]), 'GPU', r.get('GPU'))"
echo "multinode_check done rc=$?"
