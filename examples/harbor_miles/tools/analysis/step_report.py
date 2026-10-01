#!/usr/bin/env python3
"""Per-step timing report of a harbor_miles run (where does a step's wall time go, and does the trainer fit in memory?).

usage: step_report.py RUN_SPEC [JOBID ...] [--trainer-gpus 0-3 (colocated: 0-7)] [--joblog PATH ...]

Per step: end time, train_wait (rollout-side wait), train time, train tok/s, mean response tokens, raw reward (trained batch
and unfiltered), overlong rate (harbor/overlong_rate, else from the trials), sessions finished since the previous step,
peak GPU memory on the trainer GPUs (gpu-<jobid>.csv nvidia-smi samples from the head node) during the train window (from
the step's last rank-0 `Timer actor_train start` to its step line). JOBIDs restrict job logs, trials-<jobid>.jsonl and
gpu-<jobid>.csv to those jobs (default: all jobs of the run, e.g. every chunk of a chain).

Ported from QWEN27B step_report.py (2026-10-01); generalized to RUN_SPEC (local or remote) and several jobs. Change: the
train-start time of step n is the latest `actor_train start` before step n's line (was the n-th start of the job, wrong for
resumed chunks).
"""
from __future__ import annotations

import argparse
import bisect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402

TRAIN_START_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*rank00000\] timer.py:\d+ - Timer actor_train start")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="LOCAL_DIR | CLUSTER:RUN_NAME | CLUSTER:/abs/run/dir")
    ap.add_argument("jobids", nargs="*", help="restrict to these job ids (default: all jobs of the run)")
    ap.add_argument("--trainer-gpus", default="0-3", help="head-node GPU index range of the trainer (default 0-3)")
    ap.add_argument("--joblog", nargs="+", help="local job log(s) to use instead of the run's job logs")
    a = ap.parse_args()
    lo, _, hi = a.trainer_gpus.partition("-")
    lo, hi = int(lo), int(hi or lo)

    run = hm.resolve_run(a.run)
    jobs = a.jobids or run.jobids()
    text = hm.read_joblogs(run, jobids=a.jobids or None, local_paths=a.joblog)
    jl = hm.parse_joblog(text)
    met, ts = jl.metrics, jl.step_t
    starts = sorted(hm.parse_ts(m.group(1)) for line in text.splitlines()
                    if "actor_train start" in line and (m := TRAIN_START_RE.search(line)))
    trials = hm.read_trials(run, jobids=a.jobids or None, include_infra=False)
    gpu = []
    for p in run.files(r"gpu-(\d+)\.csv"):
        if not a.jobids or os.path.basename(p)[4:-4] in a.jobids:
            gpu += hm.parse_gpu_csv(run.read_text(p))

    prev_t = 0.0
    print(f"{run.cluster or 'local'} {','.join(jobs) or '-'} {run.name}: {len(ts)} steps, {len(trials)} sessions finished")
    print("| step | end | train_wait min | train min | train tok/s | resp tok | reward (trained / unfilt) | overlong | sessions | peak GB trainer GPUs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted(ts):
        d = met.get(n, {})
        end = ts[n]
        w = [t for t in trials if prev_t < (t.get("t_end") or 0) <= end]
        prev_t = end
        ol = sum(hm.is_overlong(t) for t in w) / len(w) if w else float("nan")
        i = bisect.bisect_right(starts, end)
        st = starts[i - 1] if i else None
        pk = max((m for (t, g, m) in gpu if st and st <= t <= end and lo <= g <= hi), default=0) / 1024
        print(f"| {n} | {hm.fmt_ts(end)} | {hm.num(d, 'perf/train_wait_time') / 60:.1f} | {hm.num(d, 'perf/train_time') / 60:.1f} | "
              f"{hm.num(d, 'perf/actor_train_tok_per_s'):.0f} | {hm.num(d, 'rollout/response_lengths'):.0f} | "
              f"{hm.num(d, 'rollout/raw_reward'):.3f} / {hm.num(d, 'rollout/raw_reward_unfiltered'):.3f} | "
              f"{hm.num(d, 'harbor/overlong_rate', ol):.2f} | {len(w)} | {pk:.1f} |")
    tail = [t for t in trials if (t.get("t_end") or 0) > prev_t]
    if tail:
        print(f"since last step: {len(tail)} sessions, reward {sum(float(t.get('reward') or 0) for t in tail) / len(tail):.3f}, "
              f"out tok {sum(t.get('n_output_tokens') or 0 for t in tail) / len(tail):.0f}")


if __name__ == "__main__":
    main()
