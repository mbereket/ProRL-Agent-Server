#!/usr/bin/env python3
"""Per-step and per-window task success of a harbor_miles run (did training move success, task by task?).

usage: per_task_success.py RUN_SPEC [--joblog PATH ...] [--windows 0-1,8-12,13-19,20-26] [--split train|eval|all]
                           [--per-task]

Success: "spec" = 0 for exit_status SequenceLengthLimitExceeded (a truncated session gets no credit), else the trial
reward; "raw" = the trial reward. Rows with infra_failure are excluded; --split filters rows by `split` (missing = train;
default train).
Step assignment: a trial belongs to step n if its t_end is in (t_step[n-1], t_step[n]], t_step = time of the trainer's
`log_utils ... - step N: {...}` job-log line; trials after the last step line are in progress (step = last + 1). In-flight
trials finish across step boundaries, so a step's trials are the mix the trainer saw while waiting for step n (policy lag ~
staleness), not an exact per-policy attribution.
Per-step lines: trials, tasks, task-balanced spec / raw success (mean over tasks of the per-task mean), overlong rate and
mean output tokens over trials, train/ppo_kl and rollout/raw_reward_unfiltered from the job log.
Windows (inclusive step ranges; default: 0-1 and the last 5 logged steps): the FIRST window is the base; every later window
prints task-balanced means plus the paired per-task delta of spec success vs the base (mean +- SE over tasks seen in both
windows) and "k/n tasks up". --per-task: each task's base -> last-window spec success (trials) and overlong rate.

Ported from DIAG overfit_steps.py + overfit_paired.py (2026-10-01); generalized to any run spec, split and windows. Change:
per-step spec/raw are task-balanced (were pooled over trials), and the job log's perf lines are parsed too (the unfiltered
reward lives there, so DIAG's `unfilt` column was always nan).
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402


def tb(rows, f) -> float:
    """Task-balanced mean of f over rows (each task weighted equally)."""
    by = hm.by_task(rows)
    return st.mean(st.mean(f(r) for r in v) for v in by.values()) if by else float("nan")


def out_tok(r) -> float:
    return r.get("n_output_tokens") or 0


def raw(r) -> float:
    return float(r.get("reward") or 0.0)


def summ(rows, label, base=None):
    by = hm.by_task(rows)
    if not rows:
        print(f"{label:34s} (no trials)")
        return by
    out = (f"{label:34s} n={len(rows):4d} tasks={len(by):3d} spec={tb(rows, hm.spec_success):.3f} raw={tb(rows, raw):.3f} "
           f"overlong={tb(rows, hm.is_overlong):.2f} out_tok={tb(rows, out_tok):.0f}")
    if base is not None:
        diffs = [st.mean(hm.spec_success(r) for r in by[k]) - st.mean(hm.spec_success(r) for r in base[k]) for k in by if k in base]
        if diffs:
            out += (f"  paired d(spec) {st.mean(diffs):+.3f}±{st.pstdev(diffs) / len(diffs) ** .5:.3f} "
                    f"({sum(d > 0 for d in diffs)}/{len(diffs)} tasks up)")
        else:
            out += "  paired d(spec): no task in common with the base window"
    print(out)
    return by


def parse_windows(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(","):
        lo, _, hi = part.strip().partition("-")
        out.append((int(lo), int(hi or lo)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="LOCAL_DIR | CLUSTER:RUN_NAME | CLUSTER:/abs/run/dir")
    ap.add_argument("--joblog", nargs="+", help="local job log(s) to use instead of the run's job logs")
    ap.add_argument("--windows", help="comma-separated inclusive step ranges; the first is the base (default: 0-1,<last 5>)")
    ap.add_argument("--split", default="train", choices=("train", "eval", "all"))
    ap.add_argument("--per-task", action="store_true")
    a = ap.parse_args()

    run = hm.resolve_run(a.run)
    jl = hm.parse_joblog(hm.read_joblogs(run, local_paths=a.joblog))
    ts, met = jl.step_t, jl.metrics
    rows = hm.assign_steps(hm.read_trials(run, split=a.split, include_infra=False), ts)
    print(f"{run.key}: {len(rows)} {a.split} trials (infra excluded), {len(hm.by_task(rows))} tasks, {len(ts)} steps logged")
    if not ts:
        print("no `log_utils ... step N` lines in the job logs: every trial counts as in progress", file=sys.stderr)

    for n in sorted(ts):
        w = [r for r in rows if r["step"] == n]
        if not w:
            continue
        d = met.get(n, {})
        print(f"step {n}: trials={len(w)} tasks={len(hm.by_task(w))} spec={tb(w, hm.spec_success):.3f} raw={tb(w, raw):.3f} "
              f"overlong={st.mean(hm.is_overlong(r) for r in w):.2f} out_tok={st.mean(out_tok(r) for r in w):.0f} "
              f"ppo_kl={hm.num(d, 'train/ppo_kl'):.1e} unfilt={hm.num(d, 'rollout/raw_reward_unfiltered'):.3f}")
    last = max(ts) if ts else -1
    tail = [r for r in rows if r["step"] > last]
    if tail:
        print(f"in-progress: trials={len(tail)} tasks={len(hm.by_task(tail))} spec={tb(tail, hm.spec_success):.3f}")

    if a.windows:
        wins = parse_windows(a.windows)
    else:
        wins = [(0, 1)] + ([(max(2, last - 4), last)] if last >= 2 else [])
    print()
    sel = lambda lo, hi: [r for r in rows if lo <= r["step"] <= hi]  # noqa: E731
    b = summ(sel(*wins[0]), f"base: steps {wins[0][0]}-{wins[0][1]}")
    for lo, hi in wins[1:]:
        summ(sel(lo, hi), f"steps {lo}-{hi}", b)

    if a.per_task:
        if len(wins) < 2:
            print("--per-task needs a second window")
            return
        (blo, bhi), (llo, lhi) = wins[0], wins[-1]
        E, L = hm.by_task(sel(blo, bhi)), hm.by_task(sel(llo, lhi))
        print(f"\nper task: base steps {blo}-{bhi} -> steps {llo}-{lhi}: spec(trials), overlong")
        for k in sorted(E, key=str):
            e, l = E[k], L.get(k, [])
            if l:
                print(f"  {str(k):30s} {st.mean(hm.spec_success(r) for r in e):.2f}({len(e)}) -> "
                      f"{st.mean(hm.spec_success(r) for r in l):.2f}({len(l)})   overlong "
                      f"{st.mean(hm.is_overlong(r) for r in e):.2f} -> {st.mean(hm.is_overlong(r) for r in l):.2f}")
        missing = sorted((str(k) for k in E if k not in L))
        if missing:
            print(f"  (no trials in steps {llo}-{lhi}: {', '.join(missing)})")


if __name__ == "__main__":
    main()
