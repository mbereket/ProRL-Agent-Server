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
_ROLLOUT_PERF_RE = re.compile(r"rollout_executor\] metrics\.py:\d+ - perf (\d+): (\{.*\})\s*$")
_TS_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
_PREFILL_RE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
_DECODE_RE = re.compile(r"Decode batch, #running-req: (\d+), #full token: (\d+), full token usage: ([\d.]+)"
                        r"(?:, mamba num: (\d+), mamba usage: ([\d.]+))?.*gen throughput \(token/s\): ([\d.]+)")
_STEP_LINE_RE = re.compile(r"log_utils\.py:\d+ - step (\d+): ")


def parse_engine_stats(path: str) -> dict:
    """Prefix-cache hit and KV usage from SGLang logs, bucketed by training step (the window
    between consecutive `step N:` log lines) and overall."""
    buckets: dict[int, dict] = collections.defaultdict(lambda: collections.defaultdict(float))
    step = -1
    with open(path, errors="replace") as f:
        for line in f:
            m = _STEP_LINE_RE.search(line)
            if m:
                step = int(m.group(1))
                continue
            b = buckets[step + 1]  # stats observed while producing data for the next step
            m = _PREFILL_RE.search(line)
            if m:
                b["prefill_new"] += int(m.group(2))
                b["prefill_cached"] += int(m.group(3))
                continue
            m = _DECODE_RE.search(line)
            if m:
                b["decode_lines"] += 1
                b["running_sum"] += int(m.group(1))
                b["kv_usage_max"] = max(b["kv_usage_max"], float(m.group(3)))
                if m.group(5):
                    b["mamba_usage_max"] = max(b["mamba_usage_max"], float(m.group(5)))
                b["gen_tok_s_sum"] += float(m.group(6))
    out = {}
    for k, b in sorted(buckets.items()):
        tot = b["prefill_new"] + b["prefill_cached"]
        out[k] = {
            "cache_hit": round(b["prefill_cached"] / tot, 3) if tot else None,
            "prefill_new_M": round(b["prefill_new"] / 1e6, 2),
            "kv_usage_max": round(b["kv_usage_max"], 2),
            "mamba_usage_max": round(b["mamba_usage_max"], 2),
            "avg_running_per_engine": round(b["running_sum"] / b["decode_lines"], 1) if b["decode_lines"] else None,
            "decode_tok_s_per_engine": round(b["gen_tok_s_sum"] / b["decode_lines"], 1) if b["decode_lines"] else None,
        }
    return out


def _parse_dict(text: str) -> dict:
    try:
        return ast.literal_eval(text)
    except Exception:
        return {}


def parse_log(path: str) -> dict[int, dict]:
    steps: dict[int, dict] = collections.defaultdict(dict)
    with open(path, errors="replace") as f:
        for line in f:
            for rx, prefix in ((_STEP_RE, "train"), (_ROLLOUT_RE, "rollout"), (_PERF_RE, "perf"),
                               (_ROLLOUT_PERF_RE, "rperf")):
                m = rx.search(line)
                if m and ("log_utils" in line or "model.py" in line or "train_metric_utils" in line
                          or "rollout_executor" in line):
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
    print("step  raw_reward  resp_len  total_len  wait_s  train_s  wsync_s  kl_tr_ro  stale")
    for s in sorted(steps):
        d = steps[s]
        def g(k, fmt="{:.3f}"):
            v = d.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) else "-"
        print(f"{s:4d}  {g('rollout/raw_reward'):>10}  {g('rollout/response_lengths','{:.0f}'):>8}  "
              f"{g('rollout/total_lengths','{:.0f}'):>9}  {g('perf/train_wait_time','{:.0f}'):>6}  "
              f"{g('perf/train_time','{:.0f}'):>7}  {g('perf/update_weights_time','{:.1f}'):>7}  "
              f"{g('train/train_rollout_kl','{:.1e}'):>8}  "
              f"{g('rollout/fully_async/avg_staleness','{:.2f}'):>5}")
    for s in sorted(steps):
        drops = {k.split("drop_", 1)[1]: v for k, v in steps[s].items() if k.startswith("rollout/dynamic_filter/drop_")}
        if drops:
            kept = steps[s].get("rollout/fully_async/queue_size")
            print(f"  step {s} dynamic-filter drops (groups): {drops}  unfiltered reward "
                  f"{steps[s].get('rollout/raw_reward_unfiltered')}")
    evals = {}
    with open(a.log, errors="replace") as f:
        for line in f:
            m = re.search(r"\] metrics\.py:\d+ - eval (\d+): (\{.*\})\s*$", line)
            if m:
                d = _parse_dict(m.group(2))
                evals[int(m.group(1))] = {k: v for k, v in d.items() if "/" in k and k.count("/") == 1}
    for k, v in sorted(evals.items()):
        print(f"  eval after rollout {k}: {v}")
    rewards = [steps[s]["rollout/raw_reward"] for s in sorted(steps) if "rollout/raw_reward" in steps[s]]
    summary: dict = {"n_steps": len(rewards)}
    if rewards:
        k = max(1, min(5, len(rewards) // 3))
        summary.update(first_k_mean_reward=round(statistics.mean(rewards[:k]), 3),
                       last_k_mean_reward=round(statistics.mean(rewards[-k:]), 3), k=k)
    summary["eval"] = evals
    summary["trials"] = summarize_trials(a.trials)
    # TITO session divergence: v1 one-turn rollbacks (logged) and rejected requests (4xx to the agent).
    div = collections.Counter()
    with open(a.log, errors="replace") as f:
        for line in f:
            if "Rolling back session" in line:
                div["rollbacks"] += 1
            elif "MessageValidationError" in line or "rollback failed" in line:
                div["rejected"] += 1
            elif "/v1/chat/completions HTTP/1.1\" 200" in line or "/v1/chat/completions HTTP/1.1\" 4" in line:
                div["engine_chat_calls"] += 1
    summary["session_divergence"] = dict(div)
    eng = parse_engine_stats(a.log)
    summary["engine_by_step"] = eng
    allnew = sum(v["prefill_new_M"] for v in eng.values())
    print("engine stats by step (cache_hit, kv_usage_max, avg running/engine):")
    for k, v in eng.items():
        print(f"  step {k}: hit {v['cache_hit']}  kv_max {v['kv_usage_max']}  mamba_max {v['mamba_usage_max']}  "
              f"running/engine {v['avg_running_per_engine']}  decode tok/s/engine {v['decode_tok_s_per_engine']}  "
              f"prefill_new {v['prefill_new_M']}M")
    print(json.dumps(summary, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"steps": steps, "summary": summary}, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
