"""Turn the rendered topology's single gateway node into one node per sandbox host.

    python expand_gateway_nodes.py <topology.yaml> <ip> [<ip> ...]

The template's first gateway node is the prototype: every listed host gets a
copy with id node-01, node-02, ... and public_url pointing at that host. The
bind host and port stay as rendered (0.0.0.0 on multi-node), so the same file
serves the rollout server on the head and `polar serve_gateway --node-id
node-NN` on each host. Sandboxes run on the host whose gateway accepted them.
"""
from __future__ import annotations

import copy
import sys
from urllib.parse import urlsplit, urlunsplit

import yaml


def main(path: str, ips: list[str]) -> None:
    with open(path) as f:
        topo = yaml.safe_load(f)
    proto = topo["gateway"]["nodes"][0]
    url = urlsplit(proto["public_url"])
    nodes = []
    for i, ip in enumerate(ips, 1):
        node = copy.deepcopy(proto)
        node["id"] = f"node-{i:02d}"
        node["public_url"] = urlunsplit(url._replace(netloc=f"{ip}:{url.port}"))
        nodes.append(node)
    topo["gateway"]["nodes"] = nodes
    with open(path, "w") as f:
        yaml.safe_dump(topo, f, sort_keys=False, default_flow_style=False, width=200)
    print(f"gateway nodes: " + ", ".join(f"{n['id']}@{urlsplit(n['public_url']).netloc}" for n in nodes))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: expand_gateway_nodes.py <topology.yaml> <ip> [<ip> ...]")
    main(sys.argv[1], sys.argv[2:])
