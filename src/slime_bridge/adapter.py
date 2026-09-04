"""Convert Polar rollout results into Slime samples.

Every trace in ``Trajectory.traces`` becomes one Slime ``Sample``.  All
samples produced from the same session share ``Sample.group_id`` so Slime
0.3.0's loss reducer counts the trajectory once even when it fans out into
multiple trace samples.  Builders own trace curation and per-token loss masks
— the adapter does not infer trainable positions from bridge details. Traces
that lack training tokens are dropped and represented as fully masked samples
so callers can keep the rest of the group trainable.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import re
from typing import Any, TYPE_CHECKING

from slime_bridge._messages import messages_to_text

if TYPE_CHECKING:
    from polar.rollout.models import SessionResult
    from polar.trajectory.models import Trace

logger = logging.getLogger(__name__)


class RolloutLogprobError(ValueError):
    """Raised when a trainable Polar trace lacks aligned rollout logprobs."""


OVERLONG_POLICIES = ("zero_reward_train", "drop")

# Error text a harness or inference engine emits when the conversation no
# longer fits the model's context window.
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context[ _-]?overflow|context[ _-]?length|context[ _-]?window|maximum context"
    r"|exceeds? the (?:maximum|model)|input (?:length|tokens?) .*exceed|too many tokens"
    r"|prompt is too long|input is too long|token limit|max_tokens_exceeded|length_exceeded",
    re.IGNORECASE,
)


def session_result_to_samples(
    result: "SessionResult",
    group_index: int,
    *,
    trajectory_index: int,
    reward_key: str = "score",
    max_tokens: int | None = None,
    timeout_reward_zero: bool = False,
    group_id_scope: str = "trajectory",
    overlong_policy: str = "zero_reward_train",
) -> list[Any]:
    """Convert one Polar session result into Slime samples — one per trace.

    ``timeout_reward_zero`` turns an agent TIMEOUT with captured traces into a
    trainable COMPLETED trajectory with reward 0 (the policy is then penalized
    for running out of budget instead of being masked). ``group_id_scope``
    selects slime's loss-aggregation unit: ``trajectory`` (default; every
    trajectory weighs the same) or ``prompt`` (token-mean within the prompt's
    n_samples trajectories, i.e. SkyRL's ``prompt_mean``).

    ``overlong_policy`` decides what happens to attempts that ran out of
    context. Under ``zero_reward_train`` (default) they train with reward 0
    and status COMPLETED: a trace longer than ``max_tokens`` has its response
    side truncated to fit (tokens, loss mask and rollout logprobs stay
    aligned; the prompt is kept whole), a session whose error text reports a
    context overflow keeps its traces trainable, and a trace whose final
    completion stopped on ``finish_reason == "length"`` keeps its TRUNCATED
    status but gets reward 0. Under ``drop`` over-long traces are dropped,
    overflowed sessions stay masked, and length-stopped traces keep the
    evaluator reward. Affected samples carry ``polar.overlong`` metadata.

    Every usable trace becomes an independent Sample sharing the same
    ``group_id`` key. Slime's loss reducer then averages all trace
    contributions as one trajectory, while the reward post-processor can still
    assign each trace its own advantage.

    Traces with empty tokens (or, under ``drop``, exceeding ``max_tokens``) are
    dropped (logged). If *all* traces are dropped we emit a single
    zero-gradient placeholder so Slime's flattener doesn't crash on an empty
    list and the rest of the group can still train.
    """
    if overlong_policy not in OVERLONG_POLICIES:
        raise ValueError(f"overlong_policy must be one of {OVERLONG_POLICIES}, got {overlong_policy!r}")
    Sample = _load_sample_type()
    traces = result.trajectory.traces
    group_id = group_index if group_id_scope == "prompt" else trajectory_index
    samples: list[Any] = []
    for trace_index, trace in enumerate(traces):
        sample = _build_sample(
            Sample=Sample,
            result=result,
            trace=trace,
            trace_index=trace_index,
            group_index=group_index,
            index=trajectory_index,
            group_id=group_id,
            reward_key=reward_key,
            max_tokens=max_tokens,
            timeout_reward_zero=timeout_reward_zero,
            overlong_policy=overlong_policy,
        )
        if sample is not None:
            samples.append(sample)

    if samples:
        return samples

    logger.warning(
        "Session %s: no usable trace (traces=%d, max_tokens=%s); emitting dummy placeholder",
        result.session_id, len(traces), max_tokens,
    )
    return [_build_dummy_sample(
        Sample=Sample,
        result=result,
        group_index=group_index,
        index=trajectory_index,
        group_id=group_id,
        reward_key=reward_key,
    )]


def _build_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    trace: "Trace",
    trace_index: int,
    group_index: int,
    index: int,
    group_id: int,
    reward_key: str,
    max_tokens: int | None = None,
    timeout_reward_zero: bool = False,
    overlong_policy: str = "zero_reward_train",
) -> Any | None:
    prompt_ids = list(trace.prompt_ids)
    response_ids = list(trace.response_ids)

    if not prompt_ids or not response_ids:
        logger.warning(
            "Dropping trace %d from session %s: missing tokens (prompt=%d, response=%d)",
            trace_index, result.session_id, len(prompt_ids), len(response_ids),
        )
        return None

    zero_reward_train = overlong_policy == "zero_reward_train"
    original_total_len = len(prompt_ids) + len(response_ids)
    original_response_len = len(response_ids)
    overlong_reason: str | None = None
    if max_tokens is not None and original_total_len > max_tokens:
        keep = max_tokens - len(prompt_ids)
        if not zero_reward_train or keep <= 0:
            logger.warning(
                "Dropping trace %d from session %s: total_len=%d > max_tokens=%d%s",
                trace_index, result.session_id, original_total_len, max_tokens,
                "" if not zero_reward_train else " (prompt alone does not fit)",
            )
            return None
        response_ids = response_ids[:keep]
        overlong_reason = "max_tokens"
    elif zero_reward_train and _is_context_overflow(result):
        overlong_reason = "context_overflow"
    elif zero_reward_train and trace.finish_reason == "length":
        overlong_reason = "length_stop"

    prompt_messages = deepcopy(trace.prompt_messages)
    response_messages = deepcopy(trace.response_messages)
    response_text = messages_to_text(response_messages)

    status = _sample_status(Sample, result, trace)
    reward_value = _reward_value(trace)
    if timeout_reward_zero and status is Sample.Status.ABORTED and _is_timeout(result):
        status = Sample.Status.COMPLETED
        reward_value = 0.0
    if overlong_reason is not None:
        # The attempt did not finish inside the context budget: it trains as a
        # failed attempt regardless of what the evaluator saw.
        reward_value = 0.0
        if overlong_reason != "length_stop":
            status = Sample.Status.COMPLETED

    trainable = status not in (Sample.Status.ABORTED, Sample.Status.FAILED)
    loss_mask = _loss_mask_from_trace(
        trace,
        original_response_len,
        require_loss_mask=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )[: len(response_ids)]
    if status in (Sample.Status.ABORTED, Sample.Status.FAILED):
        loss_mask = [0] * len(response_ids)
    response_log_probs = _extract_rollout_log_probs(
        trace,
        response_len=original_response_len,
        loss_mask=loss_mask + [0] * (original_response_len - len(loss_mask)),
        require_trainable_logprobs=trainable,
        session_id=result.session_id,
        trace_index=trace_index,
    )[: len(response_ids)]

    prompt_value = prompt_messages if prompt_messages else ""

    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": trace_index,
        "trace_metadata": deepcopy(getattr(trace, "metadata", {}) or {}),
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "overlong": overlong_reason is not None,
        "overlong_reason": overlong_reason,
        "original_total_len": original_total_len,
        "original_response_len": original_response_len,
        # Preserved for the longest-trace wandb artifact dump; training reads
        # tokens+logprobs, not these.
        "trace_debug": {
            "finish_reason": trace.finish_reason,
            "response_messages": deepcopy(response_messages),
        },
    }
    polar_metadata.update(_scheduler_metadata(result, trace))

    return Sample(
        group_index=group_index,
        index=index,
        prompt=prompt_value,
        tokens=prompt_ids + response_ids,
        response=response_text,
        response_length=len(response_ids),
        group_id=group_id,
        reward={reward_key: reward_value},
        loss_mask=loss_mask,
        rollout_log_probs=response_log_probs,
        status=status,
        session_id=result.session_id,
        metadata={"polar": polar_metadata},
    )


def _build_dummy_sample(
    *,
    Sample: Any,
    result: "SessionResult",
    group_index: int,
    index: int,
    group_id: int,
    reward_key: str,
) -> Any:
    """Fully masked placeholder for a session with no usable trace.

    This carries no policy, TIS, or KL contribution. It lets the scheduler
    accept a partially usable group while still surfacing empty sessions in
    Polar metrics.
    """
    polar_metadata: dict[str, Any] = {
        "node_id": result.node_id,
        "result_metadata": deepcopy(getattr(result, "metadata", {}) or {}),
        "result_error": result.error,
        "session_id": result.session_id,
        "session_status": result.status,
        "task_id": result.task_id,
        "timing": result.timing.model_dump(mode="python"),
        "trace_index": -1,
        "trajectory_error": result.trajectory.error,
        "trajectory_metadata": deepcopy(result.trajectory.metadata),
        "trajectory_status": result.trajectory.status,
        "placeholder": True,
    }
    polar_metadata.update(_scheduler_metadata(result, None))
    return Sample(
        group_index=group_index,
        index=index,
        prompt="",
        tokens=[0, 0],
        response="",
        response_length=1,
        group_id=group_id,
        reward={reward_key: 0.0},
        loss_mask=[0],
        rollout_log_probs=[0.0],
        status=Sample.Status.ABORTED,
        remove_sample=True,
        session_id=result.session_id,
        metadata={"polar": polar_metadata},
    )


def _reward_value(trace: "Trace") -> float:
    """Read the reward the evaluator already placed on the trace.

    Reward assignment is the evaluator's job (including any broadcasting
    from session-level outcomes). slime_bridge just consumes what's there.
    """
    return float(trace.reward) if trace.reward is not None else 0.0


def _scheduler_metadata(result: "SessionResult", trace: "Trace | None") -> dict[str, Any]:
    keys = {"group_id", "policy_version", "rollout_step"}
    merged: dict[str, Any] = {}
    for source in (
        getattr(result, "metadata", None),
        getattr(result.trajectory, "metadata", None),
        getattr(trace, "metadata", None) if trace is not None else None,
    ):
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source:
                merged[key] = source[key]
    return merged


def _is_timeout(result: "SessionResult") -> bool:
    return result.trajectory.status == "TIMEOUT" or result.status == "TIMEOUT"


def _is_context_overflow(result: "SessionResult") -> bool:
    """True when the session's error text or metadata reports a context overflow."""
    texts: list[str] = []
    for value in (result.error, result.trajectory.error):
        if value:
            texts.append(str(value))
    for source in (getattr(result, "metadata", None), getattr(result.trajectory, "metadata", None)):
        if isinstance(source, dict):
            for key, value in source.items():
                if isinstance(value, str) and ("error" in str(key).lower() or "reason" in str(key).lower()):
                    texts.append(value)
    return any(_CONTEXT_OVERFLOW_RE.search(text) for text in texts)


def _sample_status(Sample: Any, result: "SessionResult", trace: "Trace") -> Any:
    trajectory_status = result.trajectory.status
    if trajectory_status == "TIMEOUT" or result.status == "TIMEOUT":
        return Sample.Status.ABORTED
    if trajectory_status == "ERROR" or result.status == "ERROR" or result.error or result.trajectory.error:
        return Sample.Status.FAILED
    if trace.finish_reason == "length":
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _extract_rollout_log_probs(
    trace: "Trace",
    *,
    response_len: int,
    loss_mask: list[int],
    require_trainable_logprobs: bool,
    session_id: str,
    trace_index: int,
) -> list[float]:
    logprobs = trace.response_logprobs
    if not logprobs:
        if require_trainable_logprobs and any(loss_mask):
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing rollout_log_probs "
                "for trainable response tokens"
            )
        return [0.0] * response_len

    if len(logprobs) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: rollout_log_probs length "
            f"{len(logprobs)} != response length {response_len}"
        )

    # response_logprobs is one float per response token (interstitials are 0.0,
    # masked out by loss_mask); the builder guarantees trainable tokens carry
    # their real sampled logprob.
    return [float(value) for value in logprobs]


def _loss_mask_from_trace(
    trace: "Trace",
    response_len: int,
    *,
    require_loss_mask: bool,
    session_id: str,
    trace_index: int,
) -> list[int]:
    """Read and validate the builder-assigned per-response-token loss mask."""
    mask = list(trace.loss_mask)
    if not mask:
        if require_loss_mask:
            raise RolloutLogprobError(
                f"Session {session_id} trace {trace_index}: missing loss_mask"
            )
        return [0] * response_len
    if len(mask) != response_len:
        raise RolloutLogprobError(
            f"Session {session_id} trace {trace_index}: loss_mask length "
            f"{len(mask)} != response length {response_len}"
        )
    return [1 if int(value) else 0 for value in mask]


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to convert Polar rollouts into training samples. "
            "Ensure the Slime package is installed in the current environment."
        ) from exc
    return Sample
