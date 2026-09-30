"""Reward, reward post-process and rollout metrics for Harbor trials under Miles.

  --custom-rm-path rollout.reward_func
  --custom-reward-post-process-path rollout.post_process_rewards
  --rollout-function-path rollout.RolloutFn          (sync training; logs agent metrics)

Rewards come from the Harbor verifier via agent_function.run (sample.metadata).

Policies (env, read in the rollout process):
  HM_OVERLONG_REWARD  "zero" (default) | "verifier": reward of a trial that ran out of
                      context (SequenceLengthLimitExceeded). "verifier" keeps whatever
                      partial credit the verifier gave; "zero" trains it as a failure.
  (timeouts always score what the verifier gave, i.e. usually 0)

Infrastructure failures (agent_function sets metadata["infra_failure"]) are
excluded from the loss (``remove_sample``) AND from the group baseline: the
GRPO mean/std are taken over the group's valid trajectories only and the
excluded ones get advantage 0. Training a platform failure as reward 0 would
push the policy away from whatever it was doing when a sandbox died.

The normalization otherwise matches Miles' default (one shared reward per
rollout/trajectory, so a v2 session's several samples count once; mean over
the prompt group; optional std with --grpo-std-normalization).
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any

import torch

from miles.ray.rollout.train_data_conversion import _reward_group_segments
from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _reward_of(sample: Sample) -> float:
    md = sample.metadata or {}
    reward = float(md.get("reward", 0.0) or 0.0)
    if md.get("exit_status") == "SequenceLengthLimitExceeded" and os.environ.get("HM_OVERLONG_REWARD", "zero") == "zero":
        return 0.0
    return reward


async def reward_func(args, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(samples, list):
        return [_reward_of(s) for s in samples]
    return _reward_of(samples)


def _flatten(samples):
    out = []
    for s in samples:
        out.extend(s if isinstance(s, list) else [s])
    return out


def post_process_rewards(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    samples = _flatten(samples)
    raw = [float(s.get_reward_value(args)) for s in samples]
    normalized = [0.0] * len(samples)
    grpo_like = args.advantage_estimator in ("grpo", "gspo", "reinforce_plus_plus_baseline")
    if not (grpo_like and args.rewards_normalization):
        for s in samples:
            if (s.metadata or {}).get("infra_failure"):
                s.remove_sample = True
        return raw, list(raw)

    for segments in _reward_group_segments(args, samples, None):
        by_rollout: dict[Any, list[int]] = {}
        for i in segments:
            s = samples[i]
            key = s.rollout_id if s.rollout_id is not None else (s.index if s.index is not None else ("row", i))
            by_rollout.setdefault(key, []).append(i)
        valid_keys, valid_rewards = [], []
        for key, rows in by_rollout.items():
            infra = any((samples[i].metadata or {}).get("infra_failure") for i in rows)
            if infra:
                for i in rows:
                    samples[i].remove_sample = True
                continue
            valid_keys.append(key)
            valid_rewards.append(raw[rows[0]])
        if not valid_rewards:
            continue
        r = torch.tensor(valid_rewards, dtype=torch.float)
        adv = r - r.mean()
        if args.advantage_estimator in ("grpo", "gspo") and args.grpo_std_normalization and len(r) > 1:
            std = r.std()
            if std > 0:
                adv = adv / (std + 1e-6)
        for key, a in zip(valid_keys, adv.tolist(), strict=True):
            for i in by_rollout[key]:
                normalized[i] = a
    return raw, normalized


def aggregate_metrics(samples: list[Sample]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not samples:
        return metrics
    mds = [s.metadata or {} for s in samples]
    statuses = Counter(md.get("exit_status", "?") for md in mds)
    n = len(samples)
    for status, count in statuses.items():
        metrics[f"harbor/exit/{status.replace(' ', '_').replace(':', '')}"] = count / n
    metrics["harbor/infra_failure_rate"] = sum(bool(md.get("infra_failure")) for md in mds) / n
    agent = [md.get("agent_metrics") or {} for md in mds]
    for key in ("total_time", "env_setup_time", "agent_setup_time", "agent_run_time", "eval_time", "turns",
                "dispatch_attempts"):
        vals = [float(a[key]) for a in agent if isinstance(a.get(key), (int, float))]
        if vals:
            metrics[f"harbor/{key}_mean"] = sum(vals) / len(vals)
            metrics[f"harbor/{key}_max"] = max(vals)
    servers = Counter(a.get("agent_server") for a in agent if a.get("agent_server"))
    if servers:
        metrics["harbor/servers_used"] = float(len(servers))
        metrics["harbor/max_server_share"] = max(servers.values()) / n
    return metrics


class RolloutFn(InferenceRolloutFn):
    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        output = await super()._call_train(input)
        extra = aggregate_metrics(_flatten(output.samples))
        if extra:
            output.metrics = {**(output.metrics or {}), **extra}
            logger.info("harbor metrics for rollout %s: %s", input.rollout_id, extra)
        return output
