"""Summarize a harbor_miles run: learning curve, trial outcomes, throughput.

    python analyze_run.py <job.log> [<trials.jsonl> ...] [--out summary.json] [--per-step steps.csv]

From the Miles job log: per-step `rollout/raw_reward` (the trained batch's mean verifier reward),
train metrics (KL, logprob diff, grad norm), timings (train_wait / train time) and fully-async
staleness. From the per-trial JSONL (HM_TRIAL_LOG): outcome mix, reward by completion time,
trial wall times, sandbox setup overhead, per-server balance.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import json
import re
import statistics
import sys

_STEP_RE = re.compile(r"step (\d+): (\{.*\})\s*$")
_ROLLOUT_RE = re.compile(r"rollout (\d+): (\{.*\})\s*$")
_PERF_RE = re.compile(r"train_metric_utils\.py:\d+ - perf (\d+): (\{.*\})\s*$")


def _parse_dict(text: str) -> dict:
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def parse_log(path: str) -> dict[int, dict]:
    steps: dict[int, dict] = collections.defaultdict(dict)
    with open(path, errors="replace") as f:
        for line in f:
            for rx, prefix in ((_STEP_RE, "train"), (_ROLLOUT_RE, "rollout"), (_PERF_RE, "perf")):
                m = rx.search(line)
                if m and ("log_utils" in line or "model.py" in line or "train_metric_utils" in line):
                    steps[int(m.group(1))].update(_parse_dict(m.group(2)))
    return dict(steps)


def summarize_trials(paths: list[str]) -> dict:
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        return {}
    rows.sort(key=lambda r: r["t_end"])
    t0 = rows[0]["t_end"] - rows[0]["wall_s"]
    n = len(rows)
    out = {
        "n_trials": n,
        "span_h": round((rows[-1]["t_end"] - t0) / 3600, 2),
        "trials_per_hour": round(n / max(1e-9, (rows[-1]["t_end"] - t0) / 3600), 1),
        "exit_status": dict(collections.Counter(r["exit_status"] for r in rows)),
        "infra_failure_rate": round(sum(r["infra_failure"] for r in rows) / n, 3),
        "wall_s_p50": statistics.median(r["wall_s"] for r in rows),
        "wall_s_p90": sorted(r["wall_s"] for r in rows)[int(0.9 * (n - 1))],
        "env_setup_s_mean": round(statistics.mean(r["env_setup_time"] or 0 for r in rows), 1),
        "agent_setup_s_mean": round(statistics.mean(r["agent_setup_time"] or 0 for r in rows), 1),
        "per_server": dict(collections.Counter(r["agent_server"] for r in rows)),
    }
    # reward by completion order, in quarters (or 8 bins when large)
    bins = 8 if n >= 160 else 4
    size = max(1, n // bins)
    out["reward_by_completion_bin"] = [
        {"bin": i, "n": len(chunk), "mean_reward": round(statistics.mean(r["reward"] for r in chunk), 3),
         "solved_frac": round(sum(r["reward"] > 0 for r in chunk) / len(chunk), 3),
         "overflow_frac": round(sum(r["exit_status"] == "SequenceLengthLimitExceeded" for r in chunk) / len(chunk), 3)}
        for i, chunk in enumerate(rows[j:j + size] for j in range(0, n, size)) if chunk
    ]
    per_task = collections.defaultdict(list)
    for r in rows:
        per_task[r["instance_id"]].append(r["reward"])
    half = n // 2
    first = collections.defaultdict(list)
    last = collections.defaultdict(list)
    for r in rows[:half]:
        first[r["instance_id"]].append(r["reward"])
    for r in rows[half:]:
        last[r["instance_id"]].append(r["reward"])
    out["per_task_first_vs_second_half"] = {
        k[9:17]: [round(statistics.mean(first[k]), 3) if first[k] else None,
                  round(statistics.mean(last[k]), 3) if last[k] else None, len(first[k]), len(last[k])]
        for k in sorted(per_task)
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("trials", nargs="*")
    p.add_argument("--out", default="")
    p.add_argument("--per-step", default="")
    a = p.parse_args()
    steps = parse_log(a.log)
    keys = ["rollout/raw_reward", "rollout/response_lengths", "rollout/total_lengths", "train/train_rollout_kl",
            "train/train_rollout_logprob_abs_diff", "train/ppo_kl", "train/pg_clipfrac", "train/grad_norm",
            "perf/train_wait_time", "perf/train_time", "perf/actor_train_time", "perf/update_weights_time",
            "rollout/fully_async/avg_staleness", "rollout/fully_async/max_staleness",
            "rollout/fully_async/stale_groups_filtered"]
    if a.per_step:
        with open(a.per_step, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + keys)
            for s in sorted(steps):
                w.writerow([s] + [steps[s].get(k, "") for k in keys])
    print("step  raw_reward  resp_len  total_len  wait_s  train_s  kl_tr_ro  stale")
    for s in sorted(steps):
        d = steps[s]
        def g(k, fmt="{:.3f}"):
            v = d.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) else "-"
        print(f"{s:4d}  {g('rollout/raw_reward'):>10}  {g('rollout/response_lengths','{:.0f}'):>8}  "
              f"{g('rollout/total_lengths','{:.0f}'):>9}  {g('perf/train_wait_time','{:.0f}'):>6}  "
              f"{g('perf/train_time','{:.0f}'):>7}  {g('train/train_rollout_kl','{:.1e}'):>8}  "
              f"{g('rollout/fully_async/avg_staleness','{:.2f}'):>5}")
    rewards = [steps[s]["rollout/raw_reward"] for s in sorted(steps) if "rollout/raw_reward" in steps[s]]
    summary: dict = {"n_steps": len(rewards)}
    if rewards:
        k = max(1, min(5, len(rewards) // 3))
        summary.update(first_k_mean_reward=round(statistics.mean(rewards[:k]), 3),
                       last_k_mean_reward=round(statistics.mean(rewards[-k:]), 3), k=k)
    summary["trials"] = summarize_trials(a.trials)
    print(json.dumps(summary, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"steps": steps, "summary": summary}, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
