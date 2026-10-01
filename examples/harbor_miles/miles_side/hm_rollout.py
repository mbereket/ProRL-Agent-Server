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


_TOKENIZER = None


def _tokenizer(args):
    """The policy tokenizer, loaded once per process (only needed to re-decode a stripped sample's response text)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from miles.utils.processing_utils import load_tokenizer

            _TOKENIZER = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        except Exception:
            logger.exception("hm nan_logprob: tokenizer load failed; stripped responses get empty text (tokens are exact)")

            class _NoDecode:
                def decode(self, ids):
                    return ""

            _TOKENIZER = _NoDecode()
    return _TOKENIZER


# Sample.status as set from the final turn's engine finish_reason (miles session merge); TRUNCATED can also come from the
# collect-time max_seq_len trim. None arrives here as NaN (codec 0003); the session server's "hm codec nonfinite" line has
# None vs float NaN and the positions.
_FINISH_REASON = {"completed": "stop|tool_calls", "truncated": "length", "aborted": "abort"}

NAN_KEYS = ("tail_samples", "tail_tokens_stripped", "tail_truncated", "tail_overlong", "mid_samples", "mid_tokens")


def quarantine_nonfinite_logprobs(samples: list[Sample], args=None, where: str = "") -> dict[str, float]:
    """Non-finite rollout logprobs (NaN/inf/None) must never reach the ratio/loss, and must not cost us trajectories.

    Root cause (DIAG): when a request hits the engine's context window, SGLang returns None/NaN placeholder logprobs on
    the TAIL of that final (context-clamped) turn. Those trajectories are the overlong ones we train on with reward 0, so
    they are kept:
      * TAIL = the longest response suffix whose tokens are all non-finite or untrainable (loss_mask 0). If it holds
        any non-finite logprob, it is cut with Miles' Sample.strip_last_output_tokens (tokens, loss mask, logprobs,
        weight-version spans, response text). The sample keeps its reward and status (nan_tail_*).
      * Anything non-finite BEFORE the tail would be a real numerical problem: the sample is excluded from the loss
        (remove_sample; its reward still counts in the group baseline) and its non-finite entries are set to 0.0, since
        Miles masks the loss multiplicatively (NaN x 0 = NaN) (nan_mid_*).
    metadata["nan_logprob"] records what was done; one log line per affected sample + one per call. Idempotent.
    """
    stats = Counter()
    for s in samples:
        lp = s.rollout_log_probs
        if not lp:
            continue
        n = len(lp)
        bad = [i for i, v in enumerate(lp) if _is_nonfinite(v)]
        if not bad:
            continue
        mask = s.loss_mask if s.loss_mask is not None else [1] * n
        t = n
        while t > 0 and (_is_nonfinite(lp[t - 1]) or not mask[t - 1]):
            t -= 1
        tail_bad = [i for i in bad if i >= t]
        mid_bad = [i for i in bad if i < t]
        status = getattr(getattr(s, "status", None), "value", getattr(s, "status", None))
        md = s.metadata if isinstance(s.metadata, dict) else {}
        # Where in the turn: the turn holding the first non-finite token starts after the last untrainable token before it.
        turn_start = max((i for i in range(bad[0]) if not mask[i]), default=-1) + 1
        turn_end = next((i for i in range(bad[0], n) if not mask[i]), n)
        info = {"response_length": n, "finish_reason": _FINISH_REASON.get(status, status),
                "first_nonfinite_from_turn_end": turn_end - bad[0], "turn_len": turn_end - turn_start,
                "nan": sum(1 for i in bad if lp[i] is not None and math.isnan(lp[i])),
                "inf": sum(1 for i in bad if lp[i] is not None and math.isinf(lp[i])),
                "none": sum(1 for i in bad if lp[i] is None),
                "tail_start": t if tail_bad else None, "tail_nonfinite": len(tail_bad),
                "stripped": 0, "mid_nonfinite": len(mid_bad), "first_mid": mid_bad[0] if mid_bad else None,
                "status": status, "overlong": _is_overlong(s), "exit_status": md.get("exit_status")}
        if tail_bad:
            strip = n - t
            if strip >= s.response_length:   # nothing finite left to train on
                mid_bad = list(bad)
                info.update(mid_nonfinite=len(mid_bad), first_mid=mid_bad[0], tail_start=None)
            else:
                s.strip_last_output_tokens(strip, _tokenizer(args))
                info["stripped"] = strip
                stats["tail_samples"] += 1
                stats["tail_tokens_stripped"] += strip
                stats["tail_truncated"] += status == "truncated"
                stats["tail_overlong"] += info["overlong"]
        if mid_bad:
            lp = list(s.rollout_log_probs)
            for i in range(len(lp)):
                if _is_nonfinite(lp[i]):
                    lp[i] = 0.0
            s.rollout_log_probs = lp
            s.remove_sample = True
            stats["mid_samples"] += 1
            stats["mid_tokens"] += len(mid_bad)
        md["nan_logprob"] = info
        s.metadata = md
        logger.warning("hm nan_logprob %s: sample index=%s instance=%s %s", where, s.index, md.get("instance_id"), info)
    if stats:
        logger.warning("hm nan_logprob %s: nan_tail_samples=%d nan_tail_tokens_stripped=%d (truncated %d, overlong %d) "
                       "nan_mid_samples=%d (removed from loss; %d tokens) of %d samples", where, stats["tail_samples"],
                       stats["tail_tokens_stripped"], stats["tail_truncated"], stats["tail_overlong"],
                       stats["mid_samples"], stats["mid_tokens"], len(samples))
    return {f"harbor/nan_{k}": float(stats[k]) for k in NAN_KEYS}


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """--custom-rollout-log-function-path hook. It runs before the step's train-data conversion in every mode, incl.
    fully async. It quarantines non-finite rollout logprobs and adds harbor/* metrics (exit statuses, overlong rate,
    agent times, nan_tail_*/nan_mid_*) to the step's `perf N:` line. It never fails the step. Returns False: Miles' logging
    still runs."""
    try:
        flat = _flatten(samples)
        extra = aggregate_metrics(flat)
        extra.update(quarantine_nonfinite_logprobs(flat, args, where=f"rollout {rollout_id}"))
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
    quarantine_nonfinite_logprobs(samples, args, where="post_process")
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
