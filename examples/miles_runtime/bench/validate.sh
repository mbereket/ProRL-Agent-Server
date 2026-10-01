#!/usr/bin/env bash
# Per-cluster runtime validation: compat, versions, GPU smoke, pidns, nested apptainer, Ray (+dashboard), port range.
MR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
bash "${MR}/setup.sh" compat manifest smoke pidns nested
bash "${MR}/ray_node.sh" -- python3 -c "import ray; ray.init(); from ray.util.state import list_nodes; print('RAY_OK', ray.cluster_resources().get('GPU'), len(list_nodes()))"
echo "RANGE $(cat /proc/sys/net/ipv4/ip_local_port_range)"
