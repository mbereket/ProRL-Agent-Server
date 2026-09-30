"""harbor_miles_grpo/internal/render.py: config -> Miles args + Polar topology."""

from __future__ import annotations

import importlib.util
import os
import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RENDER = ROOT / "examples" / "harbor_miles_grpo" / "internal" / "render.py"


@pytest.fixture()
def render(monkeypatch, tmp_path):
    monkeypatch.setenv("WORKROOT", str(tmp_path / "wr"))
    monkeypatch.setenv("DRY_RUN", "1")
    monkeypatch.setenv("RAY_HEAD_IP", "10.0.0.1")
    monkeypatch.setenv("WORKER_IPS", "10.0.0.2")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.delenv("RUN_ID", raising=False)
    spec = importlib.util.spec_from_file_location("miles_render", RENDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.end_of_turn_token_id = lambda _: 248046  # no tokenizer in unit tests
    return mod


def _config(tmp_path, **overrides) -> str:
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    cfg = {
        "name": "t",
        "tasks": {"dir": str(tasks)},
        "rollout": {"num_steps": 3, "batch_size": 4, "n_samples_per_prompt": 8},
        "cluster": {"num_nodes": 2, "actor_num_gpus": 8, "tp_size": 2},
    }
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _rendered(tmp_path) -> tuple[list[str], dict, dict]:
    run_dir = tmp_path / "wr" / "harbor_miles_grpo" / "t"
    text = (run_dir / "train_args.sh").read_text()
    body = text.split("TRAIN_ARGS=(\n", 1)[1].split("\n)\n", 1)[0]
    args = [shlex.split(line)[0] for line in body.splitlines()]
    topo = yaml.safe_load((run_dir / "topology.yaml").read_text())
    polar = yaml.safe_load((run_dir / "polar_config.yaml").read_text())
    return args, topo, polar


def _value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_lora_defaults(render, tmp_path) -> None:
    render.mode_run(render.load(_config(tmp_path)))
    args, topo, polar = _rendered(tmp_path)
    assert _value(args, "--lora-rank") == "32" and _value(args, "--megatron-to-hf-mode") == "bridge"
    assert _value(args, "--update-weight-transfer-mode") == "broadcast"
    assert "--calculate-per-token-loss" in args and "--normalize-advantages" not in args
    assert "--use-dynamic-batch-size" in args and "--qkv-format" not in args
    assert _value(args, "--global-batch-size") == "32"  # one unit per trajectory
    assert _value(args, "--sglang-router-policy") == "manual"
    assert "--load" not in args and _value(args, "--start-rollout-id") == "0"
    assert _value(args, "--rollout-num-gpus") == "8"
    nodes = topo["gateway"]["nodes"]
    assert [n["public_url"] for n in nodes] == ["http://10.0.0.1:8100", "http://10.0.0.2:8100"]
    assert all(n["inference"]["extra_body"] == {"lora_path": "miles_lora"} for n in nodes)
    assert all(n["inference"]["routing_key_header"] == "X-SMG-Routing-Key" for n in nodes)
    assert polar["polar_topology_path"].endswith("/t/topology.yaml")
    assert "train_async.py" in (tmp_path / "wr" / "harbor_miles_grpo" / "t" / "train_args.sh").read_text()


def test_full_finetune_has_no_adapter_and_loads_torch_dist(render, tmp_path) -> None:
    render.mode_run(render.load(_config(tmp_path, lora={"rank": 0}, training={"lr": "1e-6"})))
    args, topo, _ = _rendered(tmp_path)
    assert "--lora-rank" not in args
    assert _value(args, "--load").endswith("_miles_torch_dist")
    assert all(n["inference"]["extra_body"] == {} for n in topo["gateway"]["nodes"])


def test_prompt_scope_and_trajectory_mean(render, tmp_path) -> None:
    cfg = _config(tmp_path, training={"group_id_scope": "prompt", "loss_aggregation": "trajectory_mean"},
                  cluster={"router_policy": "round_robin"})
    render.mode_run(render.load(cfg))
    args, topo, _ = _rendered(tmp_path)
    assert _value(args, "--global-batch-size") == "4"
    assert "--calculate-per-token-loss" not in args
    assert "routing_key_header" not in topo["gateway"]["nodes"][0]["inference"]


def test_lora_resume_picks_newest_complete_adapter(render, tmp_path) -> None:
    save = tmp_path / "wr" / "ckpt" / "harbor_miles_grpo" / "t"
    for it, complete in ((4, True), (9, True), (12, False)):
        d = save / f"iter_{it:07d}" / "adapter"
        d.mkdir(parents=True)
        (d / "adapter_megatron_rank0.pt").touch()
        if complete:
            (d / "training_state_rank0.pt").touch()
    render.mode_run(render.load(_config(tmp_path)))
    args, _, _ = _rendered(tmp_path)
    assert _value(args, "--lora-adapter-path") == str(save / "iter_0000009" / "adapter")
    assert "--start-rollout-id" not in args


def test_unknown_keys_rejected(render, tmp_path) -> None:
    with pytest.raises(SystemExit):
        render.load(_config(tmp_path, training={"dynamic_history": True}))
