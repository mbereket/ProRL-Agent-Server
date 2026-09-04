"""Requested-session denominators, tails, throughput, occupancy and per-task eval CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from slime_bridge.config import resolve_polar_slime_config
from slime_bridge.rollout import (
    AsyncPolarRolloutWorker,
    _OccupancySampler,
    _p90,
    _per_task_eval_rows,
    _polar_extra_metrics,
    _write_per_task_eval_csv,
)


def _sample(
    session_id: str,
    reward: float,
    *,
    status: str = "COMPLETED",
    placeholder: bool = False,
    run_ms: float = 3.0,
    init_ms: float = 2.0,
    staleness: int | None = None,
    response_length: int = 0,
) -> SimpleNamespace:
    polar = {
        "session_id": session_id,
        "session_status": status,
        "placeholder": placeholder,
        "timing": {
            "register_to_init_queue_ms": 1.0,
            "init_ms": init_ms,
            "run_ms": run_ms,
            "postrun_ms": 4.0,
        },
    }
    if staleness is not None:
        polar["policy_staleness"] = staleness
    s = SimpleNamespace(reward={"score": reward}, metadata={"polar": polar})
    s.response_length = response_length
    s.loss_mask = [1] * response_length
    s.remove_sample = False
    return s


def test_requested_denominator_counts_missing_sessions_as_failures() -> None:
    samples = [_sample("a", 1.0), _sample("b", 0.0)]
    full = _polar_extra_metrics(samples, rewards=[1.0, 0.0], reward_key="score", sessions_requested=2)
    assert full["polar/sessions_requested"] == 2
    assert full["polar/sessions_missing"] == 0
    assert full["polar/success_rate_all_sessions"] == 0.5
    assert full["polar/reward_mean_all_sessions"] == 0.5
    assert full["polar/status/completed_fraction"] == 1.0
    assert "polar/status/missing_fraction" not in full

    # One requested session never came back: denominator unchanged, missing = 1.
    short = _polar_extra_metrics([samples[0]], rewards=[1.0], reward_key="score", sessions_requested=2)
    assert short["polar/sessions_requested"] == 2
    assert short["polar/sessions_all"] == 1
    assert short["polar/sessions_missing"] == 1
    assert short["polar/success_rate_all_sessions"] == 0.5
    assert short["polar/reward_mean_all_sessions"] == 0.5
    assert short["polar/status/completed_fraction"] == 0.5
    assert short["polar/status/missing_fraction"] == 0.5


def test_train_path_requested_is_observed_plus_dropped_and_missing() -> None:
    samples = [_sample("a", 1.0), _sample("b", 0.0, status="TIMEOUT", placeholder=True)]
    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 0.0],
        reward_key="score",
        extra_session_outcomes=[(1.0, "COMPLETED"), (0.0, "COMPLETED")],  # zero-variance-dropped group
        extra_missing_sessions=2,  # failed task with no results
    )
    assert metrics["polar/sessions_requested"] == 6
    assert metrics["polar/sessions_all"] == 4
    assert metrics["polar/sessions_dropped"] == 2
    assert metrics["polar/sessions_missing"] == 2
    assert metrics["polar/success_rate_all_sessions"] == 2 / 6
    assert metrics["polar/reward_mean_all_sessions"] == 2 / 6
    assert metrics["polar/status/completed_fraction"] == 3 / 6
    assert metrics["polar/status/timeout_fraction"] == 1 / 6
    assert metrics["polar/status/missing_fraction"] == 2 / 6


def test_session_phase_tails_and_staleness_max() -> None:
    samples = [
        _sample(f"s{i}", 0.0, run_ms=float(i), init_ms=float(10 * i), staleness=i) for i in range(1, 11)
    ]
    metrics = _polar_extra_metrics(samples, rewards=[0.0] * 10, reward_key="score")
    assert metrics["polar/session_ms/run_max"] == 10.0
    assert metrics["polar/session_ms/run_p90"] == _p90([float(i) for i in range(1, 11)]) == 9.0
    assert metrics["polar/session_ms/init_max"] == 100.0
    assert metrics["polar/session_ms/init_p90"] == 90.0
    assert metrics["polar/staleness/mean"] == 5.5
    assert metrics["polar/staleness/max"] == 10.0


def test_generation_throughput_metrics() -> None:
    samples = [_sample("a", 1.0, response_length=300), _sample("b", 0.0, response_length=100)]
    metrics = _polar_extra_metrics(
        samples, rewards=[1.0, 0.0], reward_key="score", rollout_wall_s=8.0, gen_active_s=6.0
    )
    assert metrics["polar/gen/response_tokens_per_s"] == 50.0
    assert metrics["polar/gen/active_fraction"] == 0.75
    no_wall = _polar_extra_metrics(samples, rewards=[1.0, 0.0], reward_key="score")
    assert "polar/gen/response_tokens_per_s" not in no_wall


def _worker_args(**overrides) -> SimpleNamespace:
    args = dict(
        polar_rollout_url="http://rollout:8080",
        polar_task_template={"agent": {"harness": "codex"}},
        polar_max_async_level=2,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        update_weights_interval=1,
        polar_callback_host="127.0.0.1",
        polar_occupancy_interval_s=0,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_worker_dropped_outcomes_and_missing_are_consumed_once() -> None:
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=SimpleNamespace())
    worker._record_dropped_sessions(2, [(1.0, "COMPLETED"), (0.0, "COMPLETED")])
    worker._record_dropped_sessions(2, [])  # failed task, no results
    outcomes, missing = worker.take_dropped_session_outcomes()
    assert outcomes == [(1.0, "COMPLETED"), (0.0, "COMPLETED")]
    assert missing == 2
    assert worker.take_dropped_session_outcomes() == ([], 0)


def test_worker_generation_clock_tracks_time_with_active_sessions(monkeypatch) -> None:
    import slime_bridge.rollout as rollout_mod

    clock = {"t": 100.0}
    monkeypatch.setattr(rollout_mod.time, "monotonic", lambda: clock["t"])
    worker = AsyncPolarRolloutWorker(_worker_args(), data_source=SimpleNamespace())
    worker.mark_step_start()
    worker._record_active_counts({}, 0)
    clock["t"] = 110.0
    worker._record_active_counts({object(): None}, 4)   # generation starts at t=110
    clock["t"] = 116.0
    worker._record_active_counts({}, 0)                  # ends at t=116 -> 6 s
    clock["t"] = 120.0
    assert worker.take_gen_active_seconds() == 6.0
    # Still-active generation at the step boundary is charged up to "now".
    worker.mark_step_start()
    worker._record_active_counts({object(): None}, 1)
    clock["t"] = 123.0
    assert worker.take_gen_active_seconds() == 3.0


def test_occupancy_sampler_writes_header_and_rows(tmp_path: Path) -> None:
    snaps = iter([
        {"rollout_id": 3, "polar/scheduler/active_groups": 2, "polar/scheduler/active_sessions": 8,
         "polar/scheduler/completed_buffer": 1, "polar/scheduler/output_queue": 0,
         "polar/scheduler/deferred_queue": 0, "polar/scheduler/requested_groups": 4},
        {"rollout_id": 3, "polar/scheduler/active_groups": 0, "polar/scheduler/active_sessions": 0},
    ])
    path = tmp_path / "run" / "occupancy.csv"
    sampler = _OccupancySampler(path, interval_s=30.0, snapshot=lambda: next(snaps))
    sampler.sample_once()
    sampler.sample_once()
    rows = list(csv.DictReader(path.open()))
    assert [r["active_sessions"] for r in rows] == ["8", "0"]
    assert [r["generation_in_flight"] for r in rows] == ["1", "0"]
    assert rows[0]["rollout_id"] == "3" and rows[0]["requested_groups"] == "4"
    assert set(_OccupancySampler.COLUMNS) == set(rows[0].keys())


def test_worker_creates_occupancy_sampler_only_with_run_dir(tmp_path: Path) -> None:
    off = AsyncPolarRolloutWorker(_worker_args(), data_source=SimpleNamespace())
    assert off._occupancy is None
    on = AsyncPolarRolloutWorker(
        _worker_args(polar_run_dir=str(tmp_path), polar_occupancy_interval_s=30),
        data_source=SimpleNamespace(),
    )
    assert on._occupancy is not None
    on._occupancy.sample_once()
    assert (tmp_path / "occupancy.csv").exists()


def test_config_run_dir_from_topology_save_dir(tmp_path: Path) -> None:
    topo = tmp_path / "topology.yaml"
    topo.write_text(
        "rollout:\n  host: 127.0.0.1\n  port: 18080\n  public_url: http://127.0.0.1:18080\n"
        f"  save_dir: {tmp_path}/runs/r1/rollout_results\n"
        "gateway:\n  rollout_server_url: http://127.0.0.1:18080\n"
        "  nodes:\n    - id: node-01\n      host: 127.0.0.1\n      port: 19000\n"
    )
    cfg = resolve_polar_slime_config(_worker_args(polar_topology_path=str(topo)))
    assert cfg.run_dir == f"{tmp_path}/runs/r1"
    assert cfg.occupancy_interval_s == 0.0
    explicit = resolve_polar_slime_config(_worker_args(polar_run_dir="/x/y"))
    assert explicit.run_dir == "/x/y"
    assert resolve_polar_slime_config(_worker_args()).occupancy_interval_s == 0.0
    assert resolve_polar_slime_config(
        _worker_args(polar_occupancy_interval_s=None)
    ).occupancy_interval_s == 0.0


def test_per_task_eval_rows_and_csv(tmp_path: Path) -> None:
    groups = [
        [SimpleNamespace(metadata={"task_id": "moto-1"}), SimpleNamespace(metadata={"task_id": "moto-1"})],
        [SimpleNamespace(metadata={"instance_id": "moto-2"}), SimpleNamespace(metadata={})],
    ]
    task_results = [SimpleNamespace(task_id="t-eval-0"), SimpleNamespace(task_id="t-eval-1")]
    outputs = [
        [_sample("a", 1.0), _sample("a", 1.0), _sample("b", 0.0)],  # two traces of a, one of b
        [_sample("c", 0.5)],                                          # second session vanished
    ]
    rows = _per_task_eval_rows(groups, task_results, outputs, reward_key="score")
    assert rows[0] == {
        "position": 0, "task_id": "t-eval-0", "source_task_id": "moto-1",
        "n_requested": 2, "n_observed": 2, "n_missing": 0, "n_success": 1, "reward_mean": 0.5,
    }
    assert rows[1]["source_task_id"] == "moto-2"
    assert rows[1]["n_missing"] == 1 and rows[1]["n_success"] == 1 and rows[1]["reward_mean"] == 0.25

    path = _write_per_task_eval_csv(str(tmp_path), "ref8/val", 7, rows)
    assert path == tmp_path / "eval" / "ref8_val" / "step_7.csv"
    read = list(csv.DictReader(path.open()))
    assert read[0]["task_id"] == "t-eval-0" and read[1]["n_missing"] == "1"
    assert _write_per_task_eval_csv(None, "ref8", 7, rows) is None
