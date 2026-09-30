"""The bridge against a Miles-shaped Sample: ``rollout_id`` is the aggregation
unit and there is no ``session_id`` field (Miles dropped Slime 0.3.0's
``group_id``/``session_id``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import _compat, adapter
from slime_bridge.adapter import session_result_to_samples
from slime_bridge.reward_post_process import post_process_rewards


@dataclass
class MilesLikeSample:
    """The Miles ``Sample`` fields the bridge touches (miles/utils/types.py)."""

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: Any = ""
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: Any = None
    reward: Any = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    remove_sample: bool = False
    status: "MilesLikeSample.Status" = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def get_reward_value(self, args) -> float:
        return self.reward if not args.reward_key else self.reward[args.reward_key]


def _trace(reward: float, response_ids: list[int]) -> Trace:
    return Trace(
        prompt_ids=[1, 2, 3],
        response_ids=response_ids,
        response_logprobs=[-0.5] * len(response_ids),
        loss_mask=[1] * len(response_ids),
        reward=reward,
        finish_reason="stop",
    )


def _result(session_id: str, traces: list[Trace]) -> SessionResult:
    return SessionResult(
        session_id=session_id,
        task_id="task-1",
        status=SessionStatus.COMPLETED,
        node_id="node-a",
        timing=SessionTiming(init_ms=1.0, run_ms=2.0, postrun_ms=3.0),
        trajectory=Trajectory(status="COMPLETED", traces=traces),
    )


@pytest.fixture(autouse=True)
def _miles_sample(monkeypatch):
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: MilesLikeSample)


def test_aggregation_field_maps_to_rollout_id() -> None:
    assert _compat.aggregation_field(MilesLikeSample) == "rollout_id"


def test_multi_trace_session_shares_one_rollout_id_and_keeps_session_in_metadata() -> None:
    result = _result("sess-a", [_trace(1.0, [4, 5]), _trace(1.0, [6])])
    samples = session_result_to_samples(result, group_index=3, trajectory_index=17)
    assert [s.rollout_id for s in samples] == [17, 17]
    assert all(s.metadata["polar"]["session_id"] == "sess-a" for s in samples)
    assert not hasattr(samples[0], "session_id")
    assert _compat.sample_session_id(samples[0]) == "sess-a"


def test_prompt_scope_uses_group_index_as_unit() -> None:
    result = _result("sess-a", [_trace(1.0, [4, 5])])
    samples = session_result_to_samples(result, group_index=3, trajectory_index=17, group_id_scope="prompt")
    assert samples[0].rollout_id == 3


def test_placeholder_sample_is_fully_masked() -> None:
    result = _result("sess-empty", [])
    [sample] = session_result_to_samples(result, group_index=0, trajectory_index=9)
    assert sample.rollout_id == 9
    assert sample.remove_sample is True
    assert sample.loss_mask == [0]
    assert sample.status is MilesLikeSample.Status.ABORTED


def test_reward_post_process_groups_traces_by_session_metadata() -> None:
    class Args:
        reward_key = "score"
        rewards_normalization = True
        advantage_estimator = "grpo"
        grpo_std_normalization = False

    samples = []
    # prompt group 0: session a (2 traces, reward 1), session b (reward 0), session c (reward 0)
    for sid, index, rewards in (("a", 0, [1.0, 1.0]), ("b", 1, [0.0]), ("c", 2, [0.0])):
        traces = [_trace(r, [7, 8]) for r in rewards]
        samples += session_result_to_samples(_result(sid, traces), group_index=0, trajectory_index=index)
    raw, adv = post_process_rewards(Args(), samples)
    assert raw == [1.0, 1.0, 0.0, 0.0]
    # leave-one-trajectory-out baseline: a -> 1 - mean(0, 0) = 1; b, c -> 0 - mean(1, 0) = -0.5
    assert adv == [1.0, 1.0, -0.5, -0.5]
