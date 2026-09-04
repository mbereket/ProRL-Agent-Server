"""overlong_policy: context-exhausted attempts train with reward 0 (or are dropped)."""

from __future__ import annotations

import pytest

from polar.rollout.models import SessionResult, SessionStatus, SessionTiming
from polar.trajectory.models import Trace, Trajectory
from slime_bridge import adapter
from slime_bridge.adapter import session_result_to_samples
from slime_bridge.rollout import _polar_extra_metrics
from tests.slime_bridge.test_adapter import FakeSample, _session_result


def _long_trace(prompt_len: int = 3, response_len: int = 6) -> Trace:
    return Trace(
        prompt_ids=list(range(1, prompt_len + 1)),
        response_ids=list(range(100, 100 + response_len)),
        loss_mask=[1, 0, 1, 1, 0, 1][:response_len],
        response_logprobs=[-0.1 * (i + 1) for i in range(response_len)],
        reward=1.0,
        finish_reason="stop",
    )


def test_overlong_trace_is_truncated_aligned_and_trains_with_zero_reward(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = _long_trace()  # total 9 tokens
    samples = session_result_to_samples(
        _session_result(trace=trace), group_index=1, trajectory_index=2, max_tokens=7,
    )
    assert len(samples) == 1
    s = samples[0]
    assert s.tokens == [1, 2, 3, 100, 101, 102, 103]
    assert s.response_length == 4
    assert s.loss_mask == [1, 0, 1, 1]
    assert s.rollout_log_probs == pytest.approx([-0.1, -0.2, -0.3, -0.4])
    assert len(s.tokens) - 3 == len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
    assert s.status == FakeSample.Status.COMPLETED
    assert s.reward == {"score": 0.0}
    assert s.remove_sample is False
    polar = s.metadata["polar"]
    assert polar["overlong"] is True
    assert polar["overlong_reason"] == "max_tokens"
    assert polar["original_total_len"] == 9
    assert polar["original_response_len"] == 6


def test_overlong_trace_is_dropped_under_drop_policy(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    samples = session_result_to_samples(
        _session_result(trace=_long_trace()), group_index=1, trajectory_index=2,
        max_tokens=7, overlong_policy="drop",
    )
    assert len(samples) == 1
    assert samples[0].metadata["polar"]["placeholder"] is True
    assert samples[0].remove_sample is True


def test_prompt_that_alone_exceeds_max_tokens_is_still_dropped(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    samples = session_result_to_samples(
        _session_result(trace=_long_trace(prompt_len=8, response_len=2)),
        group_index=1, trajectory_index=2, max_tokens=8,
    )
    assert samples[0].metadata["polar"]["placeholder"] is True


def test_trace_within_budget_is_untouched(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    s = session_result_to_samples(
        _session_result(trace=_long_trace()), group_index=1, trajectory_index=2, max_tokens=9,
    )[0]
    assert s.response_length == 6 and s.reward == {"score": 1.0}
    assert s.metadata["polar"]["overlong"] is False
    assert s.metadata["polar"]["overlong_reason"] is None


def _overflow_result(trace: Trace, *, error: str) -> SessionResult:
    return SessionResult(
        session_id="session-1", task_id="task-1", status=SessionStatus.ERROR, node_id="node-a",
        timing=SessionTiming(), error=error,
        trajectory=Trajectory(status="ERROR", error=error, traces=[trace]),
    )


@pytest.mark.parametrize("error", [
    "ContextOverflow: conversation exceeds the model context window",
    "openai.BadRequestError: context_length_exceeded",
    "Input length 140000 exceeds the maximum allowed length 131072",
])
def test_context_overflow_session_with_traces_trains_with_zero_reward(monkeypatch, error) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    result = _overflow_result(_long_trace(), error=error)

    trained = session_result_to_samples(result, group_index=1, trajectory_index=2)[0]
    assert trained.status == FakeSample.Status.COMPLETED
    assert trained.reward == {"score": 0.0}
    assert trained.loss_mask == [1, 0, 1, 1, 0, 1]
    assert trained.metadata["polar"]["overlong_reason"] == "context_overflow"

    masked = session_result_to_samples(result, group_index=1, trajectory_index=2, overlong_policy="drop")[0]
    assert masked.status == FakeSample.Status.FAILED
    assert masked.loss_mask == [0] * 6
    assert masked.metadata["polar"]["overlong"] is False


def test_plain_error_session_stays_masked(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    result = _overflow_result(_long_trace(), error="step 0 exited with code 1")
    s = session_result_to_samples(result, group_index=1, trajectory_index=2)[0]
    assert s.status == FakeSample.Status.FAILED
    assert s.metadata["polar"]["overlong"] is False


def test_length_stop_keeps_truncated_status_but_zero_reward(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    trace = _long_trace()
    trace = trace.model_copy(update={"finish_reason": "length"})
    default = session_result_to_samples(_session_result(trace=trace), group_index=1, trajectory_index=2)[0]
    assert default.status == FakeSample.Status.TRUNCATED
    assert default.reward == {"score": 0.0}
    assert default.loss_mask == [1, 0, 1, 1, 0, 1]
    assert default.metadata["polar"]["overlong_reason"] == "length_stop"

    kept = session_result_to_samples(
        _session_result(trace=trace), group_index=1, trajectory_index=2, overlong_policy="drop"
    )[0]
    assert kept.status == FakeSample.Status.TRUNCATED
    assert kept.reward == {"score": 1.0}


def test_invalid_overlong_policy_rejected(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    with pytest.raises(ValueError, match="overlong_policy"):
        session_result_to_samples(_session_result(trace=_long_trace()), group_index=1, trajectory_index=2, overlong_policy="mask")


def test_metrics_count_overlong_sessions_over_requested(monkeypatch) -> None:
    monkeypatch.setattr(adapter, "_load_sample_type", lambda: FakeSample)
    over = session_result_to_samples(_session_result(trace=_long_trace()), group_index=1, trajectory_index=0, max_tokens=7)
    ok = session_result_to_samples(_session_result(trace=_long_trace()), group_index=1, trajectory_index=1, max_tokens=9)
    ok[0].metadata["polar"]["session_id"] = "session-2"
    samples = over + ok
    metrics = _polar_extra_metrics(samples, rewards=[0.0, 1.0], reward_key="score", sessions_requested=4)
    assert metrics["polar/overlong_sessions"] == 1
    assert metrics["polar/overlong_fraction"] == 0.25
    assert metrics["polar/success_rate_all_sessions"] == 0.25
    assert metrics["polar/status/completed_fraction"] == 0.5
