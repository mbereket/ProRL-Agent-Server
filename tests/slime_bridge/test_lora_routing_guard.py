"""check_rollout_lora_routing: a LoRA trainer's rollouts must select its adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from slime_bridge import config as bridge_config


def _topology(tmp_path, extra_body: dict) -> str:
    doc = {
        "rollout": {"host": "127.0.0.1", "port": 8080, "public_url": "http://127.0.0.1:8080"},
        "gateway": {
            "nodes": [
                {
                    "id": f"node-0{i}",
                    "host": "127.0.0.1",
                    "port": 8100 + i,
                    "public_url": f"http://127.0.0.1:{8100 + i}",
                    "inference": {"base_url": "http://127.0.0.1:9000", "extra_body": extra_body},
                }
                for i in (1, 2)
            ]
        },
    }
    path = tmp_path / "topology.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def test_lora_trainer_with_adapter_routed_passes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bridge_config, "expected_rollout_lora_path", lambda args: "miles_lora")
    args = SimpleNamespace(polar_topology_path=_topology(tmp_path, {"lora_path": "miles_lora"}))
    bridge_config.check_rollout_lora_routing(args, live=False)


def test_lora_trainer_without_adapter_routing_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bridge_config, "expected_rollout_lora_path", lambda args: "miles_lora")
    args = SimpleNamespace(polar_topology_path=_topology(tmp_path, {}))
    with pytest.raises(ValueError, match="node-01: topology lora_path=None"):
        bridge_config.check_rollout_lora_routing(args, live=False)


def test_lora_trainer_without_topology_fails(monkeypatch) -> None:
    monkeypatch.setattr(bridge_config, "expected_rollout_lora_path", lambda args: "miles_lora")
    with pytest.raises(ValueError, match="polar_topology_path"):
        bridge_config.check_rollout_lora_routing(SimpleNamespace(), live=False)


def test_full_finetune_with_stray_adapter_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bridge_config, "expected_rollout_lora_path", lambda args: None)
    args = SimpleNamespace(polar_topology_path=_topology(tmp_path, {"lora_path": "miles_lora"}))
    with pytest.raises(ValueError, match="expects lora_path=None"):
        bridge_config.check_rollout_lora_routing(args, live=False)


def test_no_lora_args_means_no_adapter() -> None:
    assert bridge_config.expected_rollout_lora_path(SimpleNamespace(lora_rank=0)) is None
