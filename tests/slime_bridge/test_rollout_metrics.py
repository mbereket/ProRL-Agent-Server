from __future__ import annotations

from types import SimpleNamespace

from slime_bridge.rollout import _polar_extra_metrics


def _sample(
    session_id: str,
    reward: float,
    *,
    status: str = "COMPLETED",
    placeholder: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        reward={"score": reward},
        metadata={
            "polar": {
                "session_id": session_id,
                "session_status": status,
                "placeholder": placeholder,
                "timing": {
                    "register_to_init_queue_ms": 1.0,
                    "init_ms": 2.0,
                    "run_ms": 3.0,
                    "postrun_ms": 4.0,
                },
            },
        },
    )


def test_polar_reward_mean_completed_uses_unique_non_placeholder_sessions() -> None:
    samples = [
        _sample("completed-1", 1.0),
        _sample("completed-1", 1.0),
        _sample("completed-2", 0.0),
        _sample("timeout-1", 0.0, status="TIMEOUT", placeholder=True),
        _sample("empty-completed", 0.0, placeholder=True),
    ]
    metrics = _polar_extra_metrics(
        samples,
        rewards=[1.0, 1.0, 0.0, 0.0, 0.0],
        reward_key="score",
    )

    assert metrics["polar/reward_mean"] == 0.4
    assert metrics["polar/reward_mean_completed"] == 0.5
    assert metrics["polar/rollout_success_rate"] == 0.5


def test_polar_trajectory_shape_metrics() -> None:
    def shaped(session_id, reward, *, response_length, loss_mask, record_count, status="COMPLETED"):
        s = _sample(session_id, reward, status=status)
        s.response_length = response_length
        s.loss_mask = loss_mask
        s.remove_sample = False
        s.metadata["polar"]["trajectory_metadata"] = {"record_count": record_count}
        return s

    samples = [
        shaped("a", 1.0, response_length=4, loss_mask=[1, 1, 0, 0], record_count=2),
        shaped("a", 1.0, response_length=2, loss_mask=[1, 1], record_count=2),  # second trace, same trajectory
        shaped("b", 0.0, response_length=6, loss_mask=[1, 1, 1, 0, 0, 0], record_count=3, status="TIMEOUT"),
    ]
    metrics = _polar_extra_metrics(samples, rewards=[1.0, 1.0, 0.0], reward_key="score")

    assert metrics["polar/traj/traces_mean"] == 1.5
    assert metrics["polar/traj/response_tokens_mean"] == 6.0
    assert metrics["polar/traj/response_tokens_max"] == 6.0
    assert metrics["polar/traj/trainable_token_fraction"] == 7 / 12
    assert metrics["polar/traj/turns_mean"] == 2.5
    assert metrics["polar/traj/tokens_per_turn_mean"] == (6 / 2 + 6 / 3) / 2
    assert metrics["polar/status/completed_fraction"] == 0.5
    assert metrics["polar/status/timeout_fraction"] == 0.5
