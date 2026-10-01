"""Reward, reward post-process and rollout metrics for Harbor trials under Miles.

  --custom-rm-path hm_rollout.reward_func
  --custom-reward-post-process-path hm_rollout.post_process_rewards
  --rollout-function-path hm_rollout.RolloutFn          (sync training; logs agent metrics)

Rewards come from the Harbor verifier via hm_agent.run (sample.metadata).

Policies (env, read in the rollout process):
  HM_OVERLONG_REWARD  "zero" (default, recipe) | "verifier": reward of a trajectory that ran out of
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
    if os.environ.get("HM_OVERLONG_REWARD", "zero") == "zero" and _is_overlong(sample):
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


# Sample.status as set from the final turn's engine finish_reason (miles session merge); TRUNCATED can also come from the
# collect-time max_seq_len trim. None arrives here as NaN (codec 0003); the session server's "hm codec nonfinite" line has
# None vs float NaN and the positions.
_FINISH_REASON = {"completed": "stop|tool_calls", "truncated": "length", "aborted": "abort"}

NAN_KEYS = ("logprob_samples", "tail_samples", "tail_tokens", "tail_truncated", "tail_overlong", "mid_samples",
            "mid_tokens")


def scan_nonfinite_logprobs(samples: list[Sample]) -> tuple[dict[str, float], list[dict]]:
    """Non-finite rollout logprobs (NaN/inf/None) are an ALARM, never data: the engine served a broken policy.

    History (FINDINGS F40 -> F66): they were first read as SGLang placeholders on context-clamped final turns and
    quarantined (tail stripped, sample kept). The real cause was a NaN LoRA after every bridge-LoRA resume
    (uninitialized Adam moments, Miles patch 0004): the NaN model emits no EOS, so every turn ran to the context limit.
    Training on such a batch is never valid, so this scan only measures (nothing is modified) and
    post_process_rewards() stops the job. TAIL = the longest response suffix whose tokens are all non-finite or
    untrainable (loss_mask 0); anything non-finite before it is "mid". Returns (harbor/nan_* metrics, per-sample info).
    """
    stats = Counter()
    infos = []
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
        turn_start = max((i for i in range(bad[0]) if not mask[i]), default=-1) + 1
        turn_end = next((i for i in range(bad[0], n) if not mask[i]), n)
        info = {"index": s.index, "instance": md.get("instance_id"), "response_length": n,
                "finish_reason": _FINISH_REASON.get(status, status), "status": status, "overlong": _is_overlong(s),
                "exit_status": md.get("exit_status"), "first_nonfinite_from_turn_end": turn_end - bad[0],
                "turn_len": turn_end - turn_start,
                "nan": sum(1 for i in bad if lp[i] is not None and math.isnan(lp[i])),
                "inf": sum(1 for i in bad if lp[i] is not None and math.isinf(lp[i])),
                "none": sum(1 for i in bad if lp[i] is None),
                "tail_nonfinite": len(tail_bad), "mid_nonfinite": len(mid_bad),
                "first_mid": mid_bad[0] if mid_bad else None}
        infos.append(info)
        stats["logprob_samples"] += 1
        if tail_bad:
            stats["tail_samples"] += 1
            stats["tail_tokens"] += len(tail_bad)
            stats["tail_truncated"] += status == "truncated"
            stats["tail_overlong"] += info["overlong"]
        if mid_bad:
            stats["mid_samples"] += 1
            stats["mid_tokens"] += len(mid_bad)
    return {f"harbor/nan_{k}": float(stats[k]) for k in NAN_KEYS}, infos


def _stop_on_nonfinite_logprobs(samples: list[Sample], where: str) -> None:
    """HARD STOP: any non-finite rollout logprob fails the job (RuntimeError -> the driver exits non-zero) and stops
    the chain (RUN_DIR/chain.stop), after writing RUN_DIR/FATAL-nonfinite-logprobs-<job>.txt."""
    metrics, infos = scan_nonfinite_logprobs(samples)
    if not infos:
        return
    summary = {k.removeprefix("harbor/"): int(v) for k, v in metrics.items()}
    msg = (f"HM FATAL ({where}): non-finite rollout logprobs in {len(infos)} of {len(samples)} samples {summary}. "
           "The engines served a broken policy (e.g. NaN LoRA weights after a resume; FINDINGS F66). Not training on "
           "this batch; the job stops and the chain is stopped (RUN_DIR/chain.stop). Check the trainer log for "
           "'FATAL' / non-finite grad_norm, the newest checkpoint for NaNs, and the session server's "
           "'hm codec nonfinite' lines.")
    for info in infos[:20]:
        logger.error("hm nonfinite logprobs %s: %s", where, info)
    logger.error(msg)
    run_dir = os.path.dirname(os.environ.get("HM_TRIAL_LOG", "")) or os.environ.get("RUN_DIR", "")
    if run_dir and os.path.isdir(run_dir):
        try:
            job = os.environ.get("SLURM_JOB_ID", "local")
            with open(os.path.join(run_dir, f"FATAL-nonfinite-logprobs-{job}.txt"), "a") as f:
                f.write(msg + "\n" + "\n".join(map(str, infos)) + "\n")
            with open(os.path.join(run_dir, "chain.stop"), "a") as f:
                f.write(f"hm_rollout hard stop ({where}, job {job}): non-finite rollout logprobs\n")
        except OSError:
            logger.exception("hm_rollout: could not write the FATAL marker / chain.stop under %s", run_dir)
    raise RuntimeError(msg)


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """--custom-rollout-log-function-path hook. It runs before the step's train-data conversion in every mode, incl.
    fully async, and adds harbor/* metrics (exit statuses, overlong rate, agent times, nan_* alarms) to the step's
    `perf N:` line. It never fails the step itself: post_process_rewards() runs next and hard-stops on any non-finite
    logprob, so the perf line with the nan_* counts is printed first. Returns False: Miles' logging still runs."""
    try:
        flat = _flatten(samples)
        extra = aggregate_metrics(flat)
        nan_metrics, infos = scan_nonfinite_logprobs(flat)
        extra.update(nan_metrics)
        if infos:
            logger.error("hm nonfinite logprobs rollout %s: %d of %d samples %s (hard stop follows)", rollout_id,
                         len(infos), len(flat), nan_metrics)
        if isinstance(rollout_extra_metrics, dict):
            rollout_extra_metrics.update(extra)
        else:
            logger.info("harbor metrics for rollout %s: %s", rollout_id, extra)
    except Exception:  # metrics must never kill training; post_process_rewards does the hard stop
        logger.exception("hm_rollout.log_rollout_data failed for rollout %s (metrics only)", rollout_id)
    return False


def post_process_rewards(args, samples: list[Sample] | list[list[Sample]]) -> tuple[list[float], list[float]]:
    samples = _flatten(samples)
    # HARD STOP on any non-finite rollout logprob (no quarantine: such a batch comes from a broken policy, F66).
    _stop_on_nonfinite_logprobs(samples, where="post_process_rewards")
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
