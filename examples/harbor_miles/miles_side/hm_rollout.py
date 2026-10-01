"""Reward, reward post-process and rollout metrics for Harbor trials under Miles.

  --custom-rm-path hm_rollout.reward_func
  --custom-reward-post-process-path hm_rollout.post_process_rewards
  --rollout-function-path hm_rollout.RolloutFn          (sync training; logs agent metrics)

Rewards come from the Harbor verifier via hm_agent.run (sample.metadata).

Policies (env, read in the rollout process):
  HM_OVERLONG_REWARD  "verifier" (default) | "zero": reward of a trajectory that ran out of
                      context. The verifier still runs on the sandbox state the agent left
                      (Harbor runs it after agent errors/timeouts), so "verifier" rewards the
                      outcome only: a fix that landed before the context ran out counts. "zero"
                      additionally penalizes length. Timeouts always score the verifier result.
  A trajectory is "overlong" if the agent server reports SequenceLengthLimitExceeded or the
  session server truncated it (Sample.Status.TRUNCATED) — independent of the harness's error text.

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
import math
import os
from collections import Counter
from typing import Any

from miles.ray.rollout.train_data_conversion import _reward_group_segments
from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def _is_overlong(sample: Sample) -> bool:
    md = sample.metadata or {}
    status = getattr(sample, "status", None)
    return md.get("exit_status") == "SequenceLengthLimitExceeded" or getattr(status, "value", status) == "truncated"


def _reward_of(sample: Sample) -> float:
    md = sample.metadata or {}
    reward = float(md.get("reward", 0.0) or 0.0)
    if os.environ.get("HM_OVERLONG_REWARD", "verifier") == "zero" and _is_overlong(sample):
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


def _is_nonfinite(v) -> bool:
    return v is None or not math.isfinite(v)


def quarantine_nonfinite_logprobs(samples: list[Sample], where: str = "") -> dict[str, float]:
    """Non-finite rollout logprobs (NaN/inf/None from the engine) must never reach the ratio/loss.

    For every sample with a non-finite rollout_log_probs entry:
      * the entry is replaced by 0.0: Miles masks the loss MULTIPLICATIVELY (NaN x 0 = NaN), so a NaN on even a
        masked-out token would poison the step's loss and metrics;
      * if any such entry sits on a trainable token (loss_mask 1), the sample is excluded from the loss
        (``remove_sample``; its reward still counts in the group baseline: the verifier result is valid);
      * metadata["nan_logprob"] records where they were, and one log line per affected sample says whether they sit in
        the last assistant turn and whether the trajectory was truncated/overlong.
    Idempotent: a second call finds nothing. Returns per-call counts (also logged).
    """
    stats = Counter()
    for s in samples:
        lp = s.rollout_log_probs
        if not lp:
            continue
        bad = [i for i, v in enumerate(lp) if _is_nonfinite(v)]
        if not bad:
            continue
        mask = s.loss_mask
        trainable = [i for i in bad if mask is None or (i < len(mask) and mask[i])]
        last_untrainable = max((i for i, m in enumerate(mask or []) if not m), default=-1)
        in_last_turn = all(i > last_untrainable for i in bad)
        if not isinstance(lp, list):
            lp = list(lp)
        for i in bad:
            lp[i] = 0.0
        s.rollout_log_probs = lp
        if trainable:
            s.remove_sample = True
        md = s.metadata if isinstance(s.metadata, dict) else {}
        status = getattr(getattr(s, "status", None), "value", getattr(s, "status", None))
        info = {"tokens": len(bad), "trainable": len(trainable), "first": bad[0], "last": bad[-1], "response_len": len(lp),
                "in_last_turn": in_last_turn, "status": status, "overlong": _is_overlong(s)}
        md["nan_logprob"] = info
        s.metadata = md
        stats["samples"] += 1
        stats["removed"] += bool(trainable)
        stats["tokens"] += len(bad)
        stats["last_turn"] += in_last_turn
        stats["truncated"] += status == "truncated"
        stats["overlong"] += info["overlong"]
        logger.warning("hm nan_logprob %s: sample index=%s instance=%s %s", where, s.index, md.get("instance_id"), info)
    if stats:
        logger.warning("hm nan_logprob %s: nan_logprob_samples=%d of %d (removed %d, tokens %d, all-in-last-turn %d, "
                       "truncated %d, overlong %d)", where, stats["samples"], len(samples), stats["removed"],
                       stats["tokens"], stats["last_turn"], stats["truncated"], stats["overlong"])
    return {f"harbor/nan_logprob_{k}": float(stats[k])
            for k in ("samples", "removed", "tokens", "last_turn", "truncated", "overlong")}


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """--custom-rollout-log-function-path hook. It runs before the step's train-data conversion in every mode, incl.
    fully async. It quarantines non-finite rollout logprobs and adds harbor/* metrics (exit statuses, overlong rate,
    agent times, nan_logprob_*) to the step's `perf N:` line. It never fails the step. Returns False: Miles' logging
    still runs."""
    try:
        flat = _flatten(samples)
        extra = aggregate_metrics(flat)
        extra.update(quarantine_nonfinite_logprobs(flat, where=f"rollout {rollout_id}"))
        if isinstance(rollout_extra_metrics, dict):
            rollout_extra_metrics.update(extra)
        else:
            logger.info("harbor metrics for rollout %s: %s", rollout_id, extra)
    except Exception:  # metrics must never kill training; post_process_rewards still quarantines
        logger.exception("hm_rollout.log_rollout_data failed for rollout %s (metrics only)", rollout_id)
    return False


def post_process_rewards(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    samples = _flatten(samples)
    # Normally a no-op (the log hook already ran); guarantees NaN-free logprobs if the hook is not configured.
    quarantine_nonfinite_logprobs(samples, where="post_process")
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
        mean = sum(valid_rewards) / len(valid_rewards)
        adv = [r - mean for r in valid_rewards]
        if args.advantage_estimator in ("grpo", "gspo") and args.grpo_std_normalization and len(adv) > 1:
            std = math.sqrt(sum(a * a for a in adv) / (len(adv) - 1))  # unbiased, as torch.std
            if std > 0:
                adv = [a / (std + 1e-6) for a in adv]
        for key, a in zip(valid_keys, adv, strict=True):
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
    metrics["harbor/overlong_rate"] = sum(_is_overlong(s) for s in samples) / n
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
