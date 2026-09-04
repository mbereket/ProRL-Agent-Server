"""Slime rollout bridge for Polar-managed agent sessions.

Single entrypoint ``generate_rollout_polar_async`` routes training to a
persistent background worker and evaluation to a one-shot submit+poll batch.
Both paths speak Polar's async-only HTTP surface (``/rollout/task/submit`` +
``/rollout/task/{task_id}``).
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import json
import logging
import math
import queue
import statistics
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

from polar.rollout.models import TaskResult, TaskStatus
from slime_bridge._messages import prompt_to_instruction_text
from slime_bridge.adapter import RolloutLogprobError, session_result_to_samples
from slime_bridge.config import (
    PolarSlimeConfig,
    render_instruction,
    render_task_payload,
    resolve_polar_slime_config,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 2.0  # seconds between task-status polls (eval / no-callback path)
_CALLBACK_FALLBACK_POLL_SECONDS = 60.0  # defensive backstop for dropped callbacks
_LONGEST_TRACE_ARTIFACT_INTERVAL = 5  # dump longest trace every N rollouts


class PolarRolloutSchedulerError(RuntimeError):
    """Raised when the async Polar scheduler cannot safely make progress."""


class PolarLowCompleteAcceptFractionError(PolarRolloutSchedulerError):
    """Raised when a completed task has too few trainable completed sessions."""


class PolarZeroVarianceGroupError(PolarRolloutSchedulerError):
    """Raised when every trainable trajectory in a group has the same reward."""


# (evaluator reward, terminal session status) of a session that is not trained on.
_SessionOutcome = tuple[float, str]


def _task_result_session_outcomes(task_result: Any) -> list[_SessionOutcome]:
    """One (reward, status) per session of a task result (reward 0.0 when absent)."""
    outcomes: list[_SessionOutcome] = []
    for result in getattr(task_result, "results", None) or []:
        trajectory = getattr(result, "trajectory", None)
        metadata = getattr(trajectory, "metadata", None) or {}
        value = (metadata.get("evaluation") or {}).get("reward")
        status = getattr(result, "status", None)
        status = str(getattr(status, "value", status) or "UNKNOWN")
        outcomes.append((float(value) if isinstance(value, (int, float)) else 0.0, status))
    return outcomes


def _sample_session_outcomes(samples: list[Any], reward_key: str) -> list[_SessionOutcome]:
    """One (reward, status) per distinct session among converted samples."""
    outcomes: dict[str, _SessionOutcome] = {}
    for sample in samples:
        polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
        session_id = polar_meta.get("session_id")
        if not session_id or session_id in outcomes:
            continue
        evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
        value = evaluation.get("reward")
        if isinstance(value, (int, float)):
            reward = float(value)
        elif polar_meta.get("placeholder"):
            reward = 0.0
        else:
            reward = _extract_sample_reward(sample, reward_key)
        outcomes[session_id] = (reward, str(_sample_session_status(sample) or "UNKNOWN"))
    return list(outcomes.values())


def _with_session_outcomes(exc: BaseException, outcomes: list[_SessionOutcome]) -> BaseException:
    """Tag a group-rejection error with the group's session outcomes so a dropped
    group still counts in the all-sessions metrics."""
    exc.session_outcomes = outcomes  # type: ignore[attr-defined]
    return exc


def _p90(values: list[float]) -> float:
    """Nearest-rank 90th percentile."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]


class _OccupancySampler:
    """Background thread appending scheduler gauges to a CSV at a fixed interval.

    One row per sample: wall-clock, current rollout id, active groups/sessions,
    completed buffer, output and deferred queue sizes, requested groups, and
    whether any generation is in flight (active sessions > 0).
    """

    COLUMNS = (
        "wall_clock_utc", "rollout_id", "active_groups", "active_sessions",
        "completed_buffer", "output_queue", "deferred_queue", "requested_groups",
        "generation_in_flight",
    )

    def __init__(self, path: Path, interval_s: float, snapshot: Any) -> None:
        self._path = path
        self._interval_s = float(interval_s)
        self._snapshot = snapshot
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="polar-occupancy")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def sample_once(self) -> None:
        snap = self._snapshot()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        active_sessions = int(snap.get("polar/scheduler/active_sessions", 0))
        row = (
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            int(snap.get("rollout_id", -1)),
            int(snap.get("polar/scheduler/active_groups", 0)),
            active_sessions,
            int(snap.get("polar/scheduler/completed_buffer", 0)),
            int(snap.get("polar/scheduler/output_queue", 0)),
            int(snap.get("polar/scheduler/deferred_queue", 0)),
            int(snap.get("polar/scheduler/requested_groups", 0)),
            int(active_sessions > 0),
        )
        with open(self._path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(self.COLUMNS) + "\n")
            f.write(",".join(str(v) for v in row) + "\n")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:
                logger.exception("Occupancy sampler failed to write %s", self._path)
            self._stop.wait(self._interval_s)



@dataclass(slots=True)
class _DeferredGroup:
    group: list[Any]


@dataclass(slots=True)
class _PendingGroup:
    group_id: int
    group: list[Any]
    submitted_rollout_id: int
    policy_version: int
    session_cost: int


@dataclass(slots=True)
class _CompletedGroup:
    group_id: int
    group: list[Any]
    samples: list[Any]
    task_id: str
    submitted_rollout_id: int
    policy_version: int
    session_count: int
    completed_at: float = field(default_factory=time.monotonic)

# ---------------------------------------------------------------------------
# Global worker singleton
# ---------------------------------------------------------------------------
_global_async_worker: "AsyncPolarRolloutWorker | None" = None
_worker_lock = threading.Lock()


def get_global_async_worker(args: Any, data_source: Any) -> "AsyncPolarRolloutWorker":
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is None or not _global_async_worker.is_alive():
            logger.info("Creating new async Polar rollout worker")
            _global_async_worker = AsyncPolarRolloutWorker(args, data_source)
            _global_async_worker.start()
        return _global_async_worker


def stop_global_worker() -> None:
    global _global_async_worker
    with _worker_lock:
        if _global_async_worker is not None:
            _global_async_worker.stop()
            _global_async_worker = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _build_task_payload(
    *,
    args: Any,
    config: PolarSlimeConfig,
    group: list[Any],
    rollout_id: int,
    task_position: int,
) -> dict[str, Any]:
    first_sample = group[0]
    prompt_text = prompt_to_instruction_text(getattr(first_sample, "prompt", ""))
    instruction = render_instruction(
        args=args,
        config=config,
        sample=first_sample,
        prompt_text=prompt_text,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )
    return render_task_payload(
        args=args,
        config=config,
        sample=first_sample,
        instruction=instruction,
        rollout_id=rollout_id,
        task_position=task_position,
        num_rollouts=len(group),
    )


def _attach_scheduler_metadata(
    payload: dict[str, Any],
    *,
    group_id: int,
    policy_version: int,
    rollout_step: int,
) -> None:
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("polar task metadata must be a mapping when provided")
    payload["metadata"] = {
        **metadata,
        "group_id": group_id,
        "policy_version": policy_version,
        "rollout_step": rollout_step,
    }


async def _submit_and_wait_for_task(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    *,
    poll_interval: float = _POLL_INTERVAL,
) -> TaskResult:
    """Submit one task via the async endpoint and poll until terminal."""
    resp = await client.post(
        f"{base_url}/rollout/task/submit",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    task_id = resp.json()["task_id"]

    while True:
        await asyncio.sleep(poll_interval)
        try:
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
        except (
            httpx.HTTPStatusError,
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            logger.warning("Polling Polar task %s failed; continuing: %s", task_id, exc)
            continue
        status = TaskStatus.model_validate(status_resp.json())
        if status.status in ("completed", "failed"):
            break

    return TaskResult(
        task_id=task_id,
        status=status.status,
        results=status.results,
        result_paths=status.result_paths,
    )


def _resolve_max_tokens(args: Any) -> int | None:
    """Per-sample token cap Slime's dynamic batcher can fit on one GPU.

    Megatron asserts every sample length <= max_tokens_per_gpu * cp_size.
    Deep agent trajectories can exceed this (24-turn sessions → 80k+ tokens)
    and must be dropped before they reach the batcher.
    """
    mtpg = getattr(args, "max_tokens_per_gpu", None)
    if not mtpg:
        return None
    cp_size = int(getattr(args, "context_parallel_size", 1) or 1)
    return int(mtpg) * cp_size


def _convert_task_result_to_samples(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    group: list[Any],
    *,
    max_tokens: int | None = None,
) -> list[Any]:
    """Convert one task's session results into flat Slime samples.

    Each session → one trajectory → N traces → N samples, all tagged
    with the same ``Sample.index`` so the reward post-processor groups
    them as one trajectory.  The index is taken from the originating
    group sample at matching position, falling back to the position
    within the task result.
    """
    group_index = _group_index_for(group)
    group_samples: list[Any] = []
    for pos, session_result in enumerate(task_result.results):
        source = group[pos] if pos < len(group) else None
        traj_idx = int(getattr(source, "index", pos) if source is not None else pos)
        group_samples.extend(
            session_result_to_samples(
                session_result,
                group_index,
                trajectory_index=traj_idx,
                reward_key=config.reward_key,
                max_tokens=max_tokens,
                timeout_reward_zero=config.timeout_reward_zero,
                group_id_scope=config.group_id_scope,
                overlong_policy=config.overlong_policy,
            )
        )
    return group_samples


def _trainable_token_count(sample: Any) -> int:
    if bool(getattr(sample, "remove_sample", False)):
        return 0
    loss_mask = getattr(sample, "loss_mask", None)
    if loss_mask is None:
        return int(getattr(sample, "response_length", 0) or 0)
    return sum(1 for value in loss_mask if int(value) != 0)


def _has_trainable_tokens(samples: list[Any]) -> bool:
    return any(_trainable_token_count(sample) > 0 for sample in samples)


def _low_complete_accept_fraction_rejection_reason(
    config: PolarSlimeConfig,
    task_result: TaskResult,
    samples: list[Any],
) -> str | None:
    threshold = config.min_complete_accept_fraction
    if threshold <= 0.0:
        return None

    total_sessions = len(task_result.results)
    if total_sessions <= 0:
        return "empty task results"

    completed_trainable = _completed_trainable_session_count(task_result, samples)
    required = math.ceil(total_sessions * threshold)
    if completed_trainable >= required:
        return None

    fraction = completed_trainable / total_sessions
    return (
        f"completed trainable sessions {completed_trainable}/{total_sessions} "
        f"({fraction:.3f}) below polar_min_complete_accept_fraction={threshold:g} "
        f"(requires >= {required})"
    )


def _completed_trainable_session_count(
    task_result: TaskResult,
    samples: list[Any],
) -> int:
    trainable_session_ids: set[str] = set()
    for sample in samples:
        if _trainable_token_count(sample) <= 0:
            continue
        session_id = _sample_session_id(sample)
        if session_id:
            trainable_session_ids.add(session_id)

    count = 0
    for result in task_result.results:
        if (
            _status_value(result.status) == "COMPLETED"
            and result.session_id in trainable_session_ids
        ):
            count += 1
    return count


def _sample_session_id(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    session_id = polar_meta.get("session_id") or getattr(sample, "session_id", None)
    return str(session_id) if session_id else None


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _is_zero_trainable_error(exc: BaseException) -> bool:
    return "zero trainable tokens" in str(exc)


def _zero_variance_rejection_reason(
    config: PolarSlimeConfig,
    samples: list[Any],
) -> str | None:
    """Reject a group whose trainable trajectories all share one reward.

    Such a group has zero GRPO advantage everywhere and would only occupy a
    batch slot; the worker pulls a replacement prompt instead. One reward per
    trajectory (session), over trajectories that are trainable and not
    FAILED/ABORTED.
    """
    if not config.drop_zero_variance_groups:
        return None
    rewards_by_session: dict[str, float] = {}
    for sample in samples:
        if _trainable_token_count(sample) <= 0:
            continue
        status = getattr(getattr(sample, "status", None), "name", None) or str(getattr(sample, "status", ""))
        if status.rsplit(".", 1)[-1].upper() in ("FAILED", "ABORTED"):
            continue
        session_id = _sample_session_id(sample) or str(id(sample))
        rewards_by_session.setdefault(session_id, _extract_sample_reward(sample, config.reward_key))
    if not rewards_by_session:
        return None  # already rejected by the zero-trainable check
    values = list(rewards_by_session.values())
    if len(values) < 2:
        return f"only {len(values)} trainable trajectory; no group baseline"
    if max(values) - min(values) <= config.zero_variance_tol:
        return f"all {len(values)} trainable trajectories have reward {values[0]:g}"
    return None


def _annotate_accepted_samples(
    samples: list[Any],
    *,
    accepted_rollout_id: int,
    staleness: int,
    policy_version: int,
    scheduler_group_id: int,
) -> None:
    for sample in samples:
        metadata = getattr(sample, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            sample.metadata = metadata
        polar_meta = metadata.setdefault("polar", {})
        if not isinstance(polar_meta, dict):
            polar_meta = {}
            metadata["polar"] = polar_meta
        polar_meta.update(
            {
                "accepted_rollout_id": int(accepted_rollout_id),
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
                "scheduler_group_id": int(scheduler_group_id),
            }
        )
        train_metadata = getattr(sample, "train_metadata", None)
        if train_metadata is None:
            train_metadata = {}
            sample.train_metadata = train_metadata
        train_metadata.update(
            {
                "policy_staleness": int(staleness),
                "policy_version": int(policy_version),
            }
        )


# ---------------------------------------------------------------------------
# Persistent training worker
# ---------------------------------------------------------------------------
class AsyncPolarRolloutWorker:
    """Persistent background worker that continuously submits Polar tasks.

    Runs in its own thread with a dedicated asyncio event loop.  Pulls
    sample groups from ``data_source``, submits them to the async
    ``/rollout/task/submit`` endpoint, polls until completion, converts
    results, and pushes them into ``output_queue``.  Training loops call
    ``drain_completed()`` to collect finished groups.
    """

    def __init__(self, args: Any, data_source: Any) -> None:
        self.args = args
        self.data_source = data_source
        self.config = resolve_polar_slime_config(args)
        batch_size = int(getattr(args, "rollout_batch_size", 1) or 1)
        # Output queue is a handoff channel; the durable overflow buffer is
        # `_completed_buffer`, which is drained in bounded chunks by training.
        queue_maxsize = max(32, batch_size * self.config.max_async_level * 2)
        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=queue_maxsize)
        self.deferred_queue: queue.Queue[_DeferredGroup] = queue.Queue()
        self._completed_buffer: deque[_CompletedGroup] = deque()
        self._running = True
        self._thread: threading.Thread | None = None
        self._group_counter = 0
        self._batch_size = batch_size
        self._current_rollout_id = int(getattr(args, "start_rollout_id", 0) or 0)
        self._requested_groups = 0
        self._fatal_error: BaseException | None = None
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float] = {}
        # Outcomes of sessions in dropped groups since the last step boundary:
        # they count in the all-sessions metrics even though nothing is trained
        # on. ``_dropped_missing_sessions`` counts dropped sessions with no
        # outcome at all (failed task without results).
        self._dropped_session_outcomes: list[_SessionOutcome] = []
        self._dropped_missing_sessions = 0
        self._active_groups = 0
        self._active_sessions = 0
        self._completed_buffer_size = 0
        # Wall time with >= 1 session in flight, accumulated between step boundaries.
        self._gen_active_since: float | None = None
        self._gen_active_accum_s = 0.0
        self._occupancy: _OccupancySampler | None = None
        if self.config.run_dir and self.config.occupancy_interval_s > 0:
            self._occupancy = _OccupancySampler(
                Path(self.config.run_dir) / "occupancy.csv",
                self.config.occupancy_interval_s,
                self._occupancy_snapshot,
            )
        # Per-task callback plumbing: event fires when the rollout server POSTs
        # the terminal TaskResult to our local listener.
        # Consecutive dropped groups (task failure / zero trainable / logprob
        # error). A run whose every session fails would otherwise pull
        # replacement prompts forever; trip a fatal error instead.
        self._consecutive_drops = 0
        self._task_events: dict[str, asyncio.Event] = {}
        self._task_results: dict[str, TaskResult] = {}
        self._callback_url: str | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="polar-async-rollout")
        self._thread.start()
        if self._occupancy is not None:
            self._occupancy.start()

    def stop(self) -> None:
        self._running = False
        if self._occupancy is not None:
            self._occupancy.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- results ---------------------------------------------------------------

    def set_rollout_context(self, rollout_id: int) -> None:
        with self._state_lock:
            self._current_rollout_id = int(rollout_id)

    def request_groups(self, count: int) -> None:
        if count <= 0:
            return
        with self._state_lock:
            self._requested_groups += int(count)

    def raise_if_failed(self) -> None:
        if self._fatal_error is not None:
            raise PolarRolloutSchedulerError(str(self._fatal_error)) from self._fatal_error

    def drain_completed(
        self,
        *,
        max_groups: int,
        rollout_id: int,
    ) -> list[_CompletedGroup]:
        self.raise_if_failed()

        while True:
            try:
                self._completed_buffer.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)

        accepted: list[_CompletedGroup] = []
        while self._completed_buffer and len(accepted) < max_groups:
            completed = self._completed_buffer.popleft()
            staleness = max(0, int(rollout_id) - completed.policy_version)
            if staleness > self.config.max_off_policy_steps:
                self._inc_metric("polar/stale_groups")
                reason = (
                    f"staleness {staleness} exceeded max_off_policy_steps="
                    f"{self.config.max_off_policy_steps}"
                )
                self._inc_metric("polar/dropped_groups")
                self._inc_metric("polar/dropped_stale_groups")
                self._inc_metric("polar/dropped_sessions", completed.session_count)
                self._record_dropped_sessions(
                    completed.session_count,
                    _sample_session_outcomes(completed.samples, self.config.reward_key),
                )
                logger.warning(
                    "Dropping stale Polar group %s task=%s: %s",
                    completed.group_id,
                    completed.task_id,
                    reason,
                )
                continue

            _annotate_accepted_samples(
                completed.samples,
                accepted_rollout_id=rollout_id,
                staleness=staleness,
                policy_version=completed.policy_version,
                scheduler_group_id=completed.group_id,
            )
            accepted.append(completed)

        if accepted:
            self._mark_delivered(len(accepted))
        with self._state_lock:
            self._completed_buffer_size = len(self._completed_buffer)
        return accepted

    def queue_size(self) -> int:
        with self._state_lock:
            return (
                self.output_queue.qsize()
                + self._completed_buffer_size
                + self.deferred_queue.qsize()
            )

    def take_dropped_session_outcomes(self) -> tuple[list[_SessionOutcome], int]:
        """Outcomes of sessions in groups dropped since the previous call, plus the
        number of dropped sessions that left no outcome (consumed once)."""
        with self._state_lock:
            outcomes = self._dropped_session_outcomes
            missing = self._dropped_missing_sessions
            self._dropped_session_outcomes = []
            self._dropped_missing_sessions = 0
            return outcomes, missing

    def _record_dropped_sessions(self, session_count: int, outcomes: list[_SessionOutcome]) -> None:
        with self._state_lock:
            self._dropped_session_outcomes.extend(outcomes)
            self._dropped_missing_sessions += max(0, int(session_count) - len(outcomes))

    def mark_step_start(self) -> None:
        """Reset the in-flight generation clock at a training step boundary."""
        with self._state_lock:
            self._gen_active_accum_s = 0.0
            self._gen_active_since = time.monotonic() if self._active_sessions > 0 else None

    def take_gen_active_seconds(self) -> float:
        """Wall seconds since ``mark_step_start`` during which >= 1 session was in flight."""
        with self._state_lock:
            total = self._gen_active_accum_s
            if self._gen_active_since is not None:
                now = time.monotonic()
                total += now - self._gen_active_since
                self._gen_active_since = now
            self._gen_active_accum_s = 0.0
            return total

    def _occupancy_snapshot(self) -> dict[str, float]:
        out = self.snapshot_metrics()
        with self._state_lock:
            out["rollout_id"] = float(self._current_rollout_id)
        return out

    def snapshot_metrics(self) -> dict[str, float]:
        with self._state_lock:
            out = dict(self._metrics)
            out["polar/scheduler/active_groups"] = float(self._active_groups)
            out["polar/scheduler/active_sessions"] = float(self._active_sessions)
            out["polar/scheduler/completed_buffer"] = float(self._completed_buffer_size)
            out["polar/scheduler/output_queue"] = float(self.output_queue.qsize())
            out["polar/scheduler/deferred_queue"] = float(self.deferred_queue.qsize())
            out["polar/scheduler/requested_groups"] = float(self._requested_groups)
            return out

    # -- internal --------------------------------------------------------------

    def _run_loop(self) -> None:
        asyncio.run(self._async_loop())

    async def _async_loop(self) -> None:
        logger.info("Async Polar rollout worker started")
        active: dict[asyncio.Task[None], _PendingGroup] = {}
        active_session_cost = 0
        wakeup = asyncio.Event()

        callback_server, callback_task = await self._start_callback_listener()
        timeout = None if self.config.request_timeout is None else httpx.Timeout(self.config.request_timeout)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                while self._running:
                    done = [t for t in active if t.done()]
                    for t in done:
                        pending = active.pop(t)
                        active_session_cost -= pending.session_cost
                        try:
                            t.result()
                        except Exception as exc:
                            logger.exception("Polar async task failed")
                            self._set_fatal(exc)
                            self._running = False
                    self._record_active_counts(active, active_session_cost)

                    while self._running and self._can_admit_group(active, active_session_cost):
                        try:
                            next_group = self._next_group_for_submission()
                        except Exception as exc:
                            self._set_fatal(exc)
                            self._running = False
                            break
                        if next_group is None:
                            break
                        session_cost = len(next_group.group)
                        if session_cost > self.config.max_session_concurrency:
                            self._set_fatal(
                                PolarRolloutSchedulerError(
                                    f"Prompt group needs {session_cost} sessions but "
                                    f"derived max_session_concurrency is "
                                    f"{self.config.max_session_concurrency}"
                                )
                            )
                            self._running = False
                            break
                        if active_session_cost + session_cost > self.config.max_session_concurrency:
                            self.deferred_queue.put(next_group)
                            break

                        gid = self._group_counter
                        self._group_counter += 1
                        submitted_rollout_id, policy_version = self._rollout_context()
                        pending = _PendingGroup(
                            group_id=gid,
                            group=next_group.group,
                            submitted_rollout_id=submitted_rollout_id,
                            policy_version=policy_version,
                            session_cost=session_cost,
                        )
                        task = asyncio.create_task(
                            self._submit_and_collect(client, pending),
                            name=f"polar-rollout-task-{gid}",
                        )
                        task.add_done_callback(lambda _: wakeup.set())
                        active[task] = pending
                        active_session_cost += session_cost
                        self._record_active_counts(active, active_session_cost)

                    if self._running:
                        try:
                            await asyncio.wait_for(wakeup.wait(), timeout=0.5)
                        except asyncio.TimeoutError:
                            pass
                        wakeup.clear()

            if active:
                logger.info("Waiting for %d in-flight Polar tasks", len(active))
                await asyncio.gather(*active.keys(), return_exceptions=True)
        finally:
            callback_server.should_exit = True
            try:
                await asyncio.wait_for(callback_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Callback listener did not shut down within 5s")
        logger.info("Async Polar rollout worker stopped")

    async def _start_callback_listener(self) -> tuple[uvicorn.Server, asyncio.Task[None]]:
        """Bind a FastAPI listener for TaskResult callbacks."""
        app = FastAPI()

        @app.post("/callback/task_result")
        async def on_task_result(request: Request) -> dict[str, Any]:
            payload = await request.json()
            task_id = payload.get("task_id") if isinstance(payload, dict) else None
            if not task_id:
                return {"ok": False, "reason": "missing task_id"}
            try:
                result = TaskResult.model_validate(payload)
            except Exception:
                logger.exception("Invalid callback payload for task %s", task_id)
                return {"ok": False, "reason": "invalid payload"}
            self._task_results[task_id] = result
            event = self._task_events.get(task_id)
            if event is not None:
                event.set()
            return {"ok": True}

        config = uvicorn.Config(
            app=app, host=self.config.callback_host, port=0,
            log_level="warning", lifespan="on",
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name="polar-callback-listener")
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        self._callback_url = f"http://{self.config.callback_host}:{port}/callback/task_result"
        logger.info("Polar trainer callback listener bound to %s", self._callback_url)
        return server, task

    async def _submit_and_collect(
        self, client: httpx.AsyncClient, pending: _PendingGroup
    ) -> None:
        last_error: BaseException | None = None

        if self._running:
            try:
                completed = await self._submit_attempt(client, pending)
                self._consecutive_drops = 0
                await self._emit_completed(completed)
                return
            except Exception as exc:
                last_error = exc

        if last_error is None:
            return

        if _is_zero_trainable_error(last_error):
            category_metric = "polar/dropped_zero_trainable_groups"
            reason = "zero trainable tokens"
        elif isinstance(last_error, PolarLowCompleteAcceptFractionError):
            category_metric = "polar/dropped_low_complete_fraction_groups"
            reason = "low complete accept fraction"
        elif isinstance(last_error, PolarZeroVarianceGroupError):
            category_metric = "polar/dropped_zero_variance_groups"
            reason = "zero reward variance"
        elif isinstance(last_error, RolloutLogprobError):
            category_metric = "polar/dropped_logprob_error_groups"
            reason = "rollout logprob error"
        else:
            category_metric = "polar/dropped_failed_groups"
            reason = "task failure"

        self._inc_metric("polar/dropped_groups")
        self._inc_metric(category_metric)
        self._inc_metric("polar/dropped_sessions", pending.session_cost)
        self._record_dropped_sessions(
            pending.session_cost, list(getattr(last_error, "session_outcomes", None) or [])
        )
        logger.warning(
            "Dropping Polar group %s because of %s: %s",
            pending.group_id,
            reason,
            last_error,
        )
        # Zero-variance drops are an expected filter, not a failure.
        if not isinstance(last_error, PolarZeroVarianceGroupError):
            self._consecutive_drops += 1
            limit = self.config.max_consecutive_dropped_groups
            if limit and self._consecutive_drops >= limit:
                self._set_fatal(
                    PolarRolloutSchedulerError(
                        f"{self._consecutive_drops} consecutive Polar groups dropped "
                        f"(last: {reason}: {last_error}); stopping instead of pulling "
                        "replacement prompts forever. Raise polar_max_consecutive_dropped_groups "
                        "or set it to 0 to disable."
                    )
                )
                self._running = False
        return

    async def _submit_attempt(
        self,
        client: httpx.AsyncClient,
        pending: _PendingGroup,
    ) -> _CompletedGroup:
        payload = _build_task_payload(
            args=self.args, config=self.config, group=pending.group,
            rollout_id=pending.group_id, task_position=0,
        )
        payload["task_id"] = str(payload["task_id"])
        _attach_scheduler_metadata(
            payload,
            group_id=pending.group_id,
            policy_version=pending.policy_version,
            rollout_step=pending.submitted_rollout_id,
        )
        task_result = await self._submit_with_callback(client, payload)

        session_outcomes = _task_result_session_outcomes(task_result)
        rejection_reason = self._task_rejection_reason(task_result, pending.group)
        if rejection_reason is not None:
            raise _with_session_outcomes(PolarRolloutSchedulerError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            ), session_outcomes)

        group_samples = _convert_task_result_to_samples(
            self.config, task_result, pending.group,
            max_tokens=_resolve_max_tokens(self.args),
        )
        if not group_samples:
            raise _with_session_outcomes(PolarRolloutSchedulerError(f"Task {task_result.task_id} converted to zero samples"), session_outcomes)
        if not _has_trainable_tokens(group_samples):
            raise _with_session_outcomes(PolarRolloutSchedulerError(
                f"Task {task_result.task_id} produced zero trainable tokens"
            ), session_outcomes)
        rejection_reason = _low_complete_accept_fraction_rejection_reason(
            self.config, task_result, group_samples
        )
        if rejection_reason is not None:
            raise _with_session_outcomes(PolarLowCompleteAcceptFractionError(
                f"Task {task_result.task_id} cannot be accepted: {rejection_reason}"
            ), session_outcomes)
        rejection_reason = _zero_variance_rejection_reason(self.config, group_samples)
        if rejection_reason is not None:
            raise _with_session_outcomes(PolarZeroVarianceGroupError(
                f"Task {task_result.task_id} dropped: {rejection_reason}"
            ), session_outcomes)

        return _CompletedGroup(
            group_id=pending.group_id,
            group=pending.group,
            samples=group_samples,
            task_id=task_result.task_id,
            submitted_rollout_id=pending.submitted_rollout_id,
            policy_version=pending.policy_version,
            session_count=len(task_result.results),
        )

    async def _emit_completed(self, completed: _CompletedGroup) -> None:
        while self._running:
            try:
                self.output_queue.put_nowait(completed)
                self._inc_metric("polar/completed_groups")
                return
            except queue.Full:
                self._inc_metric("polar/output_queue_full_waits")
                await asyncio.sleep(0.1)

    def _next_group_for_submission(self) -> _DeferredGroup | None:
        try:
            deferred = self.deferred_queue.get_nowait()
            self._inc_metric("polar/deferred_queue_dequeues")
            return deferred
        except queue.Empty:
            pass

        groups = self.data_source.get_samples(1)
        if not groups:
            return None
        group = groups[0]
        if not group:
            raise PolarRolloutSchedulerError("Slime data source returned an empty sample group")
        return _DeferredGroup(group=group)

    def _can_admit_group(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> bool:
        requested_groups = self._shared_requested_groups()
        if requested_groups <= 0:
            return False
        if len(active) >= self.config.max_concurrency:
            return False
        if active_session_cost >= self.config.max_session_concurrency:
            return False
        owned_groups = (
            len(active)
            + self.output_queue.qsize()
            + self._shared_completed_buffer_size()
            + self.deferred_queue.qsize()
        )
        admission_window = min(
            requested_groups,
            self._batch_size * self.config.max_async_level,
        )
        return owned_groups < admission_window

    def _task_rejection_reason(self, task_result: TaskResult, group: list[Any]) -> str | None:
        if task_result.status != "completed":
            return f"task status={task_result.status}"
        if not task_result.results:
            return "empty task results"
        if len(task_result.results) != len(group):
            return f"session count {len(task_result.results)} != expected {len(group)}"
        return None

    def _rollout_context(self) -> tuple[int, int]:
        with self._state_lock:
            return self._current_rollout_id, self._current_rollout_id

    def _shared_requested_groups(self) -> int:
        with self._state_lock:
            return self._requested_groups

    def _mark_delivered(self, count: int) -> None:
        with self._state_lock:
            self._requested_groups = max(0, self._requested_groups - int(count))

    def _shared_completed_buffer_size(self) -> int:
        with self._state_lock:
            return self._completed_buffer_size

    def _record_active_counts(
        self,
        active: dict[asyncio.Task[None], _PendingGroup],
        active_session_cost: int,
    ) -> None:
        with self._state_lock:
            self._active_groups = len(active)
            self._active_sessions = active_session_cost
            now = time.monotonic()
            if active_session_cost > 0 and self._gen_active_since is None:
                self._gen_active_since = now
            elif active_session_cost == 0 and self._gen_active_since is not None:
                self._gen_active_accum_s += now - self._gen_active_since
                self._gen_active_since = None

    def _inc_metric(self, key: str, amount: float = 1.0) -> None:
        with self._state_lock:
            self._metrics[key] = self._metrics.get(key, 0.0) + amount

    def _set_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

    async def _submit_with_callback(
        self, client: httpx.AsyncClient, payload: dict[str, Any]
    ) -> TaskResult:
        """Submit a task, wait on its completion event, and fall back to polling."""
        task_id = payload["task_id"]
        # Register event BEFORE submit so a fast callback cannot arrive first.
        event = asyncio.Event()
        self._task_events[task_id] = event
        payload["callback_url"] = self._callback_url
        base_url = self.config.rollout_server_url
        try:
            resp = await client.post(
                f"{base_url}/rollout/task/submit",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            return await self._await_task_result(client, task_id, event)
        finally:
            self._task_events.pop(task_id, None)
            self._task_results.pop(task_id, None)

    async def _await_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
        event: asyncio.Event,
    ) -> TaskResult:
        """Wait on the completion event with a defensive 60s fallback poll."""
        base_url = self.config.rollout_server_url
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=_CALLBACK_FALLBACK_POLL_SECONDS)
            except asyncio.TimeoutError:
                status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
                status_resp.raise_for_status()
                status = TaskStatus.model_validate(status_resp.json())
                if status.status in ("completed", "failed"):
                    return TaskResult(
                        task_id=task_id, status=status.status,
                        results=status.results, result_paths=status.result_paths,
                    )
                continue
            result = self._task_results.get(task_id)
            if result is not None:
                return result
            # Race: event set but result missing — re-poll once.
            status_resp = await client.get(f"{base_url}/rollout/task/{task_id}")
            status_resp.raise_for_status()
            status = TaskStatus.model_validate(status_resp.json())
            return TaskResult(
                task_id=task_id, status=status.status,
                results=status.results, result_paths=status.result_paths,
            )


# ---------------------------------------------------------------------------
# One-shot eval rollout
# ---------------------------------------------------------------------------
async def _run_eval_rollout(
    args: Any,
    rollout_id: int,
    data_source: Any,
) -> Any:
    config = resolve_polar_slime_config(args)
    eval_datasets = list(getattr(args, "eval_datasets", []) or [])
    if eval_datasets:
        data: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for dataset_cfg in eval_datasets:
            dataset_name, dataset_data, dataset_metrics = await _run_eval_dataset(
                args=args,
                config=config,
                rollout_id=rollout_id,
                dataset_cfg=dataset_cfg,
            )
            data[dataset_name] = dataset_data
            metrics.update(_prefix_eval_metrics(dataset_name, dataset_metrics))

        RolloutFnEvalOutput = _load_rollout_eval_output_type()
        return RolloutFnEvalOutput(data=data, metrics=metrics)

    logger.warning(
        "Polar eval called without args.eval_datasets; falling back to the training data source. "
        "Pass --eval-prompt-data to evaluate validation prompts."
    )
    sample_groups = _pull_sample_groups(data_source, args.rollout_batch_size)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=config.eval_dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    RolloutFnEvalOutput = _load_rollout_eval_output_type()
    return RolloutFnEvalOutput(
        data={config.eval_dataset_name: dataset_data},
        metrics=metrics,
    )


async def _run_eval_dataset(
    *,
    args: Any,
    config: PolarSlimeConfig,
    rollout_id: int,
    dataset_cfg: Any,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    dataset_name = str(getattr(dataset_cfg, "name", "") or config.eval_dataset_name)
    sample_groups = _load_eval_sample_groups(args, dataset_cfg)
    dataset_data, metrics = await _submit_eval_groups(
        args=args,
        config=config,
        dataset_name=dataset_name,
        rollout_id=rollout_id,
        sample_groups=sample_groups,
    )
    return dataset_name, dataset_data, metrics


async def _submit_eval_groups(
    *,
    args: Any,
    config: PolarSlimeConfig,
    dataset_name: str,
    rollout_id: int,
    sample_groups: list[list[Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not sample_groups:
        raise ValueError("Polar eval dataset produced no sample groups")

    timeout = None if config.request_timeout is None else httpx.Timeout(config.request_timeout)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def _run_one(position: int, group: list[Any]) -> TaskResult:
        async with semaphore:
            payload = _build_task_payload(
                args=args, config=config, group=group,
                rollout_id=rollout_id, task_position=position,
            )
            payload["task_id"] = _eval_task_id(
                payload["task_id"],
                dataset_name=dataset_name,
                rollout_id=rollout_id,
                position=position,
            )
            _attach_scheduler_metadata(
                payload,
                group_id=position,
                policy_version=rollout_id,
                rollout_step=rollout_id,
            )
            return await _submit_and_wait_for_task(client, config.rollout_server_url, payload)

    eval_start = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        task_results = await asyncio.gather(
            *(_run_one(pos, g) for pos, g in enumerate(sample_groups))
        )
    eval_wall_s = time.monotonic() - eval_start

    output_groups: list[list[Any]] = []
    max_tokens = _resolve_max_tokens(args)
    for group, task_result in zip(sample_groups, task_results, strict=True):
        output_groups.append(
            _convert_task_result_to_samples(
                config, task_result, group,
                max_tokens=max_tokens,
            )
        )

    metrics = _build_metrics(
        config,
        task_results,
        output_groups,
        reward_filter="completed",
        sessions_requested=sum(len(group) for group in sample_groups),
        rollout_wall_s=eval_wall_s,
    )
    csv_path = _write_per_task_eval_csv(
        config.run_dir, dataset_name, rollout_id,
        _per_task_eval_rows(sample_groups, task_results, output_groups, config.reward_key),
    )
    if csv_path is not None:
        logger.info("Wrote per-task eval results for %s to %s", dataset_name, csv_path)
    flat_samples = [sample for group in output_groups for sample in group]
    reward_samples = _completed_session_samples(flat_samples)

    return {
        "rewards": [_extract_sample_reward(s, config.reward_key) for s in reward_samples],
        "all_rewards": [_extract_sample_reward(s, config.reward_key) for s in flat_samples],
        "truncated": [_is_truncated(s) for s in reward_samples],
        "all_truncated": [_is_truncated(s) for s in flat_samples],
        "samples": flat_samples,
    }, metrics


def _eval_task_id(base_task_id: Any, *, dataset_name: str, rollout_id: int, position: int) -> str:
    """Namespace eval task ids away from train task ids.

    Training ids commonly use ``{rollout_id}-{sample.group_index}``; eval uses
    ``position`` as group index, so eval 11 / item 11 would collide with train
    group 11. A suffix keeps task polling and persisted result dirs separate.
    """
    safe_dataset = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_name
    )
    return f"{base_task_id}-eval-{safe_dataset}-{rollout_id}-{position}"


def _completed_session_samples(samples: list[Any]) -> list[Any]:
    return [
        sample for sample in samples
        if _sample_session_status(sample) == "COMPLETED"
        and not bool(
            (getattr(sample, "metadata", {}) or {})
            .get("polar", {})
            .get("placeholder")
        )
    ]


def _sample_session_status(sample: Any) -> str | None:
    polar_meta = (getattr(sample, "metadata", {}) or {}).get("polar", {})
    status = polar_meta.get("session_status")
    return getattr(status, "value", status)


def _load_eval_sample_groups(args: Any, dataset_cfg: Any) -> list[list[Any]]:
    Sample = _load_sample_type()
    path = str(getattr(dataset_cfg, "path"))
    input_key = getattr(dataset_cfg, "input_key", None) or getattr(args, "input_key", "prompt")
    label_key = getattr(dataset_cfg, "label_key", None) or getattr(args, "label_key", None)
    metadata_key = getattr(dataset_cfg, "metadata_key", None) or getattr(args, "metadata_key", "metadata")
    tool_key = getattr(dataset_cfg, "tool_key", None) or getattr(args, "tool_key", None)
    group_size = int(
        getattr(dataset_cfg, "n_samples_per_eval_prompt", None)
        or getattr(args, "n_samples_per_eval_prompt", None)
        or 1
    )
    if group_size <= 0:
        raise ValueError("n_samples_per_eval_prompt must be positive")

    groups: list[list[Any]] = []
    sample_index = 0
    for prompt_index, row in enumerate(_read_jsonl_rows(path)):
        if input_key not in row:
            raise KeyError(f"Eval row {prompt_index} in {path} missing input key {input_key!r}")

        metadata = _inject_eval_metadata(dataset_cfg, row.get(metadata_key))
        if tool_key and tool_key in row:
            tools = row[tool_key]
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata["tools"] = tools

        group: list[Any] = []
        for _ in range(group_size):
            sample = Sample(
                prompt=copy.deepcopy(row[input_key]),
                label=row.get(label_key) if label_key else None,
                metadata=copy.deepcopy(metadata),
                group_index=prompt_index,
                index=sample_index,
            )
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            group.append(sample)
            sample_index += 1
        groups.append(group)

    return groups


def _read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Eval row {line_number} in {path} is not a JSON object")
            rows.append(row)
    return rows


def _inject_eval_metadata(dataset_cfg: Any, sample_metadata: Any) -> dict[str, Any]:
    inject = getattr(dataset_cfg, "inject_metadata", None)
    if callable(inject):
        metadata = inject(sample_metadata)
    elif isinstance(sample_metadata, dict):
        metadata = dict(sample_metadata)
    else:
        metadata = {}
    return metadata


def _prefix_eval_metrics(dataset_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    prefixed: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("polar/"):
            prefixed[f"polar/eval/{dataset_name}/{key.removeprefix('polar/')}"] = value
        else:
            prefixed[f"polar/eval/{dataset_name}/{key}"] = value
    return prefixed


def _pull_sample_groups(data_source: Any, batch_size: int) -> list[list[Any]]:
    getter = getattr(data_source, "get_samples", None)
    if callable(getter):
        groups = getter(batch_size)
    elif callable(data_source):
        groups = data_source(batch_size)
    else:
        raise ValueError("data_source must expose get_samples(num_samples) or be callable")
    if not isinstance(groups, list):
        raise ValueError("data_source.get_samples must return a list of sample groups")
    for group in groups:
        if not group:
            raise ValueError("Slime data source returned an empty sample group")
    return groups


def _build_metrics(
    config: PolarSlimeConfig,
    task_results: list[TaskResult],
    output_groups: list[list[Any]],
    *,
    reward_filter: str = "all",
    sessions_requested: int | None = None,
    rollout_wall_s: float | None = None,
) -> dict[str, Any]:
    flat_samples = [sample for group in output_groups for sample in group]
    all_rewards = [_extract_sample_reward(s, config.reward_key) for s in flat_samples]
    completed_rewards = [
        _extract_sample_reward(s, config.reward_key)
        for s in _completed_session_samples(flat_samples)
    ]
    if reward_filter == "all":
        rewards = all_rewards
    elif reward_filter == "completed":
        rewards = completed_rewards
    else:
        raise ValueError("reward_filter must be 'all' or 'completed'")
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(
        flat_samples, rewards, config.reward_key,
        sessions_requested=sessions_requested,
        rollout_wall_s=rollout_wall_s,
    ))
    return metrics


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def generate_rollout_polar_async(args: Any, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    """Slime-compatible async rollout entrypoint.

    Training runs are served by a persistent background worker that pulls
    from ``data_source`` and drains completed groups on each call.
    Evaluation runs are served by a one-shot submit+poll batch over the
    same async HTTP surface.
    """
    if evaluation:
        return asyncio.run(_run_eval_rollout(args, rollout_id, data_source))

    async_worker = get_global_async_worker(args, data_source)
    async_worker.set_rollout_context(rollout_id)
    async_worker.mark_step_start()
    target = getattr(args, "rollout_batch_size", 1)
    async_worker.request_groups(int(target))

    data: list[list[Any]] = []
    start = time.monotonic()
    last_progress = start

    while len(data) < target:
        made_progress = False
        completed_groups = async_worker.drain_completed(
            max_groups=target - len(data),
            rollout_id=rollout_id,
        )
        for completed in completed_groups:
            data.append(completed.samples)
            made_progress = True

        now = time.monotonic()
        if made_progress:
            last_progress = now
        elif now - last_progress > 60:
            logger.warning(
                "No progress for 60s. Queue=%d, accepted=%d/%d",
                async_worker.queue_size(), len(data), target,
            )
            last_progress = now

        if len(data) < target:
            time.sleep(0.05)

    elapsed = time.monotonic() - start
    logger.info("Async rollout collected %d groups in %.1fs (queue=%d)", len(data), elapsed, async_worker.queue_size())

    _maybe_dump_longest_trace_artifact(rollout_id, data)

    RolloutFnTrainOutput = _load_rollout_train_output_type()
    flat = [s for g in data for s in g]
    rewards = [_extract_sample_reward(s, async_worker.config.reward_key) for s in flat]
    dropped_outcomes, dropped_missing = async_worker.take_dropped_session_outcomes()
    metrics: dict[str, Any] = {}
    metrics.update(_polar_extra_metrics(
        flat, rewards, async_worker.config.reward_key,
        extra_session_outcomes=dropped_outcomes,
        extra_missing_sessions=dropped_missing,
        rollout_wall_s=elapsed,
        gen_active_s=async_worker.take_gen_active_seconds(),
    ))
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


def _maybe_dump_longest_trace_artifact(
    rollout_id: int, data: list[list[Any]], *, interval: int = _LONGEST_TRACE_ARTIFACT_INTERVAL
) -> None:
    """Dump the longest session in this rollout's batch as a wandb artifact.

    Groups samples by ``session_id``, picks the session with the largest
    aggregated assistant tokens, and writes its full message chain (per
    trace) to a JSON artifact. Silently no-ops if wandb isn't initialized.
    """
    if interval <= 0 or rollout_id % interval != 0:
        return
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return

    by_session: dict[str, list[Any]] = {}
    for group in data:
        for sample in group:
            sid = getattr(sample, "session_id", None) or "unknown"
            by_session.setdefault(sid, []).append(sample)
    if not by_session:
        return

    def _session_tokens(samples: list[Any]) -> int:
        return sum(int(getattr(s, "response_length", 0) or 0) for s in samples)

    longest_sid, longest_samples = max(by_session.items(), key=lambda kv: _session_tokens(kv[1]))
    total_tokens = _session_tokens(longest_samples)
    if total_tokens <= 0:
        return

    longest_samples = sorted(
        longest_samples,
        key=lambda s: int((s.metadata.get("polar") or {}).get("trace_index", 0) or 0),
    )
    traces = []
    for sample in longest_samples:
        polar_meta = sample.metadata.get("polar") or {}
        trace_debug = polar_meta.get("trace_debug") or {}
        status = getattr(sample, "status", None)
        traces.append({
            "trace_index": polar_meta.get("trace_index"),
            "finish_reason": trace_debug.get("finish_reason"),
            "response_length": int(getattr(sample, "response_length", 0) or 0),
            "status": getattr(status, "value", None) if status is not None else None,
            "prompt_messages": sample.prompt if isinstance(sample.prompt, list) else [],
            "response_messages": trace_debug.get("response_messages") or [],
        })

    first = longest_samples[0]
    first_meta = first.metadata.get("polar") or {}
    reward = getattr(first, "reward", None)
    if isinstance(reward, dict):
        session_reward = float(reward.get("score", 0.0))
    elif isinstance(reward, (int, float)):
        session_reward = float(reward)
    else:
        session_reward = 0.0

    payload = {
        "rollout_id": int(rollout_id),
        "session_id": longest_sid,
        "task_id": first_meta.get("task_id"),
        "node_id": first_meta.get("node_id"),
        "total_assistant_tokens": int(total_tokens),
        "session_reward": session_reward,
        "num_traces": len(traces),
        "traces": traces,
    }

    try:
        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / f"longest_trace_r{rollout_id}.json"
            fpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            artifact = wandb.Artifact(
                name=f"longest_trace_r{rollout_id}", type="rollout-trace"
            )
            artifact.add_file(str(fpath))
            wandb.run.log_artifact(artifact)
    except Exception:
        logger.exception("Failed to log longest-trace wandb artifact")
        return

    logger.info(
        "Logged longest-trace artifact rollout=%d session=%s traces=%d tokens=%d",
        rollout_id, longest_sid, len(traces), total_tokens,
    )


def _group_index_for(group: list[Any]) -> int:
    if group and getattr(group[0], "group_index", None) is not None:
        return int(group[0].group_index)
    return -1


def _extract_sample_reward(sample: Any, reward_key: str) -> float:
    reward = getattr(sample, "reward", None)
    if isinstance(reward, dict):
        if reward_key in reward:
            return float(reward[reward_key])
        if "score" in reward:
            return float(reward["score"])
    if isinstance(reward, (int, float)):
        return float(reward)
    return 0.0


def _polar_extra_metrics(
    flat_samples: list[Any],
    rewards: list[float],
    reward_key: str,
    extra_session_outcomes: list[_SessionOutcome] | None = None,
    *,
    extra_missing_sessions: int = 0,
    sessions_requested: int | None = None,
    rollout_wall_s: float | None = None,
    gen_active_s: float | None = None,
) -> dict[str, float]:
    """Compact user-facing Polar metrics for W&B.

    ``polar/reward_mean`` averages over trained traces only. The
    ``*_all_sessions`` metrics are over *requested* sessions: every session
    with an outcome counts once — sessions in ``flat_samples`` (including
    placeholders for sessions with no trainable trace) plus
    ``extra_session_outcomes`` (sessions of groups the scheduler dropped) —
    and every requested session without an outcome counts as reward 0 /
    status ``MISSING``. ``sessions_requested`` is the known request size
    (eval: prompts x n); when ``None`` it is observed sessions plus
    ``extra_missing_sessions`` (train: dropped sessions that left no outcome).

    ``rollout_wall_s`` and ``gen_active_s`` (wall time with >= 1 session in
    flight) feed the ``polar/gen/*`` throughput metrics.
    """
    out: dict[str, float] = {}
    session_eval_reward: dict[str, float] = {}
    seen: set[str] = set()
    register_to_init_queue_ms: list[float] = []
    init_ms: list[float] = []
    run_ms: list[float] = []
    postrun_ms: list[float] = []
    session_is_placeholder: dict[str, bool] = {}
    session_report: dict[str, dict[str, Any]] = {}
    completed_session_rewards: list[float] = []
    policy_staleness: list[float] = []
    # Per-trajectory shape: traces, LLM calls (turns), response and trainable
    # tokens, terminal status.
    session_traces: dict[str, int] = {}
    session_turns: dict[str, int] = {}
    session_response_tokens: dict[str, int] = {}
    session_trainable_tokens: dict[str, int] = {}
    session_status: dict[str, str] = {}
    overlong_sessions: set[str] = set()
    for sample in flat_samples:
        polar_meta = sample.metadata.get("polar", {})
        if "policy_staleness" in polar_meta:
            policy_staleness.append(float(polar_meta["policy_staleness"]))
        session_id = polar_meta.get("session_id")
        is_placeholder = bool(polar_meta.get("placeholder"))
        if not session_id:
            continue
        if polar_meta.get("overlong"):
            overlong_sessions.add(session_id)
        if not is_placeholder:
            session_traces[session_id] = session_traces.get(session_id, 0) + 1
            session_response_tokens[session_id] = (
                session_response_tokens.get(session_id, 0)
                + int(getattr(sample, "response_length", 0) or 0)
            )
            session_trainable_tokens[session_id] = (
                session_trainable_tokens.get(session_id, 0) + _trainable_token_count(sample)
            )
        if session_id not in seen:
            seen.add(session_id)
            session_status[session_id] = str(_sample_session_status(sample) or "UNKNOWN")
            record_count = (polar_meta.get("trajectory_metadata") or {}).get("record_count")
            if isinstance(record_count, int):
                session_turns[session_id] = record_count
            timing = polar_meta.get("timing") or {}
            if timing:
                register_to_init_queue_ms.append(
                    float(timing.get("register_to_init_queue_ms", 0.0))
                )
                init_ms.append(float(timing.get("init_ms", 0.0)))
                run_ms.append(float(timing.get("run_ms", 0.0)))
                postrun_ms.append(float(timing.get("postrun_ms", 0.0)))
            session_is_placeholder[session_id] = is_placeholder
            evaluation = (polar_meta.get("trajectory_metadata") or {}).get("evaluation") or {}
            eval_reward = evaluation.get("reward")
            session_eval_reward[session_id] = (
                float(eval_reward) if isinstance(eval_reward, (int, float))
                else (0.0 if is_placeholder else _extract_sample_reward(sample, reward_key))
            )
            report = evaluation.get("report") or {}
            if isinstance(report, dict) and report:
                session_report[session_id] = report
            if _sample_session_status(sample) == "COMPLETED" and not is_placeholder:
                completed_session_rewards.append(
                    _extract_sample_reward(sample, reward_key)
                )

    if init_ms:
        out["polar/session_ms/register_to_init_queue_mean"] = (
            sum(register_to_init_queue_ms) / len(register_to_init_queue_ms)
        )
        out["polar/session_ms/init_mean"] = sum(init_ms) / len(init_ms)
        out["polar/session_ms/init_max"] = max(init_ms)
        out["polar/session_ms/init_p90"] = _p90(init_ms)
        out["polar/session_ms/run_mean"] = sum(run_ms) / len(run_ms)
        out["polar/session_ms/run_max"] = max(run_ms)
        out["polar/session_ms/run_p90"] = _p90(run_ms)
        out["polar/session_ms/postrun_mean"] = sum(postrun_ms) / len(postrun_ms)
    if rewards:
        out["polar/reward_mean"] = sum(rewards) / len(rewards)
    if len(rewards) > 1:
        out["polar/reward_std"] = statistics.pstdev(rewards)
    if completed_session_rewards:
        out["polar/reward_mean_completed"] = (
            sum(completed_session_rewards) / len(completed_session_rewards)
        )
    if policy_staleness:
        out["polar/staleness/mean"] = sum(policy_staleness) / len(policy_staleness)
        out["polar/staleness/max"] = max(policy_staleness)

    # Outcome distribution over requested sessions.
    extra = list(extra_session_outcomes or [])
    observed_rewards = list(session_eval_reward.values()) + [r for r, _ in extra]
    observed_statuses = list(session_status.values()) + [st for _, st in extra]
    observed = len(observed_rewards)
    if sessions_requested is None:
        requested = observed + max(0, int(extra_missing_sessions))
    else:
        requested = int(sessions_requested)
    missing = max(0, requested - observed)
    if requested > 0:
        out["polar/sessions_requested"] = float(requested)
        out["polar/sessions_all"] = float(observed)
        out["polar/sessions_dropped"] = float(len(extra))
        out["polar/sessions_missing"] = float(missing)
        out["polar/reward_mean_all_sessions"] = sum(observed_rewards) / requested
        out["polar/success_rate_all_sessions"] = (
            sum(1 for r in observed_rewards if r > 0) / requested
        )
        status_counts: dict[str, int] = {}
        for st in observed_statuses:
            status_counts[st] = status_counts.get(st, 0) + 1
        if missing:
            status_counts["MISSING"] = status_counts.get("MISSING", 0) + missing
        for status, count in sorted(status_counts.items()):
            out[f"polar/status/{status.lower()}_fraction"] = count / requested
        out["polar/overlong_sessions"] = float(len(overlong_sessions))
        out["polar/overlong_fraction"] = len(overlong_sessions) / requested

    total_sessions = len(seen)
    empty_sessions = sum(1 for p in session_is_placeholder.values() if p)
    if total_sessions > 0:
        out["polar/rollout_success_rate"] = (
            total_sessions - empty_sessions
        ) / total_sessions
    if session_traces:
        n = len(session_traces)
        out["polar/traj/traces_mean"] = sum(session_traces.values()) / n
        out["polar/traj/response_tokens_mean"] = sum(session_response_tokens.values()) / n
        out["polar/traj/response_tokens_max"] = float(max(session_response_tokens.values()))
        total_response = sum(session_response_tokens.values())
        if total_response > 0:
            out["polar/traj/trainable_token_fraction"] = (
                sum(session_trainable_tokens.values()) / total_response
            )
    if session_turns:
        out["polar/traj/turns_mean"] = sum(session_turns.values()) / len(session_turns)
        per_turn = [
            session_response_tokens[sid] / turns
            for sid, turns in session_turns.items()
            if turns > 0 and sid in session_response_tokens
        ]
        if per_turn:
            out["polar/traj/tokens_per_turn_mean"] = sum(per_turn) / len(per_turn)
    if session_report:
        graded_sessions = len(session_report)
        resolved = sum(1 for r in session_report.values() if r.get("resolved"))
        out["polar/eval/resolved_rate"] = resolved / graded_sessions
    if rollout_wall_s is not None and rollout_wall_s > 0:
        out["polar/gen/response_tokens_per_s"] = (
            sum(session_response_tokens.values()) / rollout_wall_s
        )
        if gen_active_s is not None:
            out["polar/gen/active_fraction"] = min(1.0, max(0.0, gen_active_s / rollout_wall_s))
    return out


def _per_task_eval_rows(
    sample_groups: list[list[Any]],
    task_results: list[Any],
    output_groups: list[list[Any]],
    reward_key: str,
) -> list[dict[str, Any]]:
    """One row per eval prompt: requested vs observed sessions and their outcomes."""
    rows: list[dict[str, Any]] = []
    for position, (group, task_result, samples) in enumerate(
        zip(sample_groups, task_results, output_groups, strict=True)
    ):
        outcomes = _sample_session_outcomes(samples, reward_key)
        source_meta = getattr(group[0], "metadata", None) if group else None
        source_task_id = ""
        if isinstance(source_meta, dict):
            for key in ("task_id", "instance_id", "id"):
                if source_meta.get(key) not in (None, ""):
                    source_task_id = str(source_meta[key])
                    break
        n_requested = len(group)
        n_observed = len(outcomes)
        rows.append({
            "position": position,
            "task_id": str(getattr(task_result, "task_id", "")),
            "source_task_id": source_task_id,
            "n_requested": n_requested,
            "n_observed": n_observed,
            "n_missing": max(0, n_requested - n_observed),
            "n_success": sum(1 for r, _ in outcomes if r > 0),
            "reward_mean": (sum(r for r, _ in outcomes) / n_requested) if n_requested else 0.0,
        })
    return rows


def _write_per_task_eval_csv(
    run_dir: str | None, dataset_name: str, rollout_id: int, rows: list[dict[str, Any]]
) -> Path | None:
    """Write ``${run_dir}/eval/<dataset>/step_<rollout_id>.csv``; ``None`` when no run dir."""
    if not run_dir or not rows:
        return None
    safe_dataset = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in dataset_name)
    path = Path(run_dir) / "eval" / safe_dataset / f"step_{int(rollout_id)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(str(row[c]) for c in columns) + "\n")
    return path


def _is_truncated(sample: Any) -> bool:
    status = getattr(sample, "status", None)
    return getattr(status, "value", status) == "truncated"


def _load_rollout_train_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnTrainOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar rollouts from a Slime trainer."
        ) from exc
    return RolloutFnTrainOutput


def _load_rollout_eval_output_type() -> Any:
    try:
        from slime.rollout.base_types import RolloutFnEvalOutput
    except ImportError as exc:
        raise ImportError(
            "Slime is required to run Polar evaluation rollouts from a Slime trainer."
        ) from exc
    return RolloutFnEvalOutput


def _load_sample_type() -> Any:
    try:
        from slime.utils.types import Sample
    except ImportError as exc:
        raise ImportError(
            "Slime is required to build Polar evaluation samples from eval datasets."
        ) from exc
    return Sample


atexit.register(stop_global_worker)
