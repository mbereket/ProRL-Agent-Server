#!/usr/bin/env python3
"""Per-step profile of a harbor_miles run (filtering, staleness, TITO, outcome mix, overlong token share, evals).

usage: profile_steps.py RUN_SPEC [--joblog PATH ...] [--jobs JOBID ...] [--group-size N]

Per step: wall s, train s, trainer idle s, weight sync s, groups kept/dropped (dynamic filter), unfiltered / trained reward,
staleness mean/max, ppo_kl, trials ended in the step window and their overlong / timeout / agent-error shares, infra
failures, TITO critical (special-token / non-assistant) / assistant-text mismatch rates, mean trained response tokens, and
the overlong share of generated tokens. Then warm steps/h, all-trial outcome summary, eval-tagged trials and one line per
eval metric line (every `eval/<name>` dataset in it).
Trials are bucketed into step n by end time in (t_step[n-1], t_step[n]] (in-flight trials finish across step boundaries, so
this is the trial mix the trainer saw while waiting for step n, not an exact per-step attribution). Eval-split trials are
reported separately (untagged rows count as train). Groups kept = rollout/num_training_samples / group size
(--group-size, else --n-samples-per-prompt from args-*.txt, else 8).

Ported from PATH-B profile_r2.py (2026-10-01); generalized to RUN_SPEC, group size from args, any eval dataset name (was
hard-coded eval/train48).
"""
from __future__ import annotations

import argparse
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402

CRIT = ("special_token_count", "special_token_type", "non_assistant_text")


def ovl_tok(rows):
    tot = sum(r.get("n_output_tokens") or 0 for r in rows)
    ovl = sum(r.get("n_output_tokens") or 0 for r in rows if hm.is_overlong(r))
    return f"{100 * ovl / tot:.0f}%" if tot else "-"


def f(d, k, fmt="{:.0f}"):
    v = d.get(k)
    return fmt.format(v) if isinstance(v, (int, float)) else "-"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="LOCAL_DIR | CLUSTER:RUN_NAME | CLUSTER:/abs/run/dir")
    ap.add_argument("--joblog", nargs="+", help="local job log(s) to use instead of the run's job logs")
    ap.add_argument("--jobs", nargs="+", help="restrict job logs and trials to these job ids")
    ap.add_argument("--group-size", type=int, help="samples per prompt (default: from args-*.txt, else 8)")
    a = ap.parse_args()

    run = hm.resolve_run(a.run)
    jl = hm.parse_joblog(hm.read_joblogs(run, jobids=a.jobs, local_paths=a.joblog))
    m_all, ts, evals = jl.metrics, jl.step_t, jl.evals
    gs = a.group_size or int(hm.arg_value(hm.read_args(run), "--n-samples-per-prompt", 8))
    trials = hm.read_trials(run, jobids=a.jobs)
    eval_trials = [r for r in trials if hm.row_split(r) == "eval"]
    trials = [r for r in trials if hm.row_split(r) != "eval"]

    print(f"{run.key}: group size {gs}")
    print("| step | wall s | train s | trainer idle s | wsync s | groups kept/dropped (drop%) | unfilt reward | trained reward |"
          " staleness mean/max | ppo_kl | trials ended | overlong | timeout | agent err | infra | TITO crit/asst mismatch |"
          " mean response tok (trained) | overlong share of generated tokens |")
    print("|" + "---|" * 18)
    prev_t = None
    for n in sorted(m_all):
        d = m_all[n]
        nd = sum(v for k, v in d.items() if k.startswith("rollout/dynamic_filter/drop_") and isinstance(v, (int, float)))
        kept = hm.num(d, "rollout/num_training_samples", 8 * gs) / gs
        drop = f"{kept:.0f}/{nd:.0f} ({100 * nd / (nd + kept):.0f}%)" if "rollout/raw_reward" in d and nd + kept else "-"
        wall = f"{ts[n] - prev_t:.0f}" if (n in ts and prev_t) else "-"
        if n in ts:
            lo = prev_t if prev_t else 0
            win = [r for r in trials if lo < r["t_end"] <= ts[n]]
        else:
            win = [r for r in trials if r["t_end"] > (prev_t or 0)]
        c = collections.Counter(r.get("exit_status") for r in win)
        N = max(len(win), 1)
        crit = sum(hm.num(d, f"rollout/tito_session_mismatch_rate/v1/{k}", 0) for k in CRIT)
        asst = d.get("rollout/tito_session_mismatch_rate/v1/assistant_text")
        print(f"| {n}{'' if n in ts else ' (in progress)'} | {wall} | {f(d, 'perf/actor_train_time')} | {f(d, 'perf/train_wait_time')} | "
              f"{f(d, 'perf/update_weights_time', '{:.1f}')} | {drop} | {f(d, 'rollout/raw_reward_unfiltered', '{:.3f}')} | "
              f"{f(d, 'rollout/raw_reward', '{:.3f}')} | {f(d, 'rollout/fully_async/avg_staleness', '{:.2f}')}/"
              f"{f(d, 'rollout/fully_async/max_staleness')} | {f(d, 'train/ppo_kl', '{:.1e}')} | {len(win)} | "
              f"{100 * c[hm.OVERLONG] / N:.0f}% | {100 * c['TimeLimitExceeded'] / N:.0f}% | "
              f"{100 * c['AgentError'] / N:.0f}% | {sum(bool(r.get('infra_failure')) for r in win)} | "
              f"{crit:.3f}/{asst if not isinstance(asst, (int, float)) else round(asst, 2)} | {f(d, 'rollout/response_len/mean')} | "
              f"{ovl_tok(win)} |")
        if n in ts:
            prev_t = ts[n]
    done = sorted(ts.items())
    if len(done) >= 3:
        warm = done[1:]
        h = (warm[-1][1] - warm[0][1]) / 3600
        if h > 0:
            print(f"\nwarm steps/h {(len(warm) - 1) / h:.1f} (steps {warm[0][0]}-{warm[-1][0]}, includes eval stalls)")
    if trials:
        c = collections.Counter(r.get("exit_status") for r in trials)
        N = len(trials)
        span = (trials[-1]["t_end"] - trials[0]["t_end"] + (trials[0].get("wall_s") or 0)) / 3600
        walls = [r["wall_s"] for r in trials if r.get("wall_s") is not None]
        print(f"all trials {N} ({N / span if span > 0 else float('nan'):.0f}/h): overlong {100 * c[hm.OVERLONG] / N:.1f}%, "
              f"timeout {100 * c['TimeLimitExceeded'] / N:.1f}%, agent err {100 * c['AgentError'] / N:.1f}%, "
              f"infra {sum(bool(r.get('infra_failure')) for r in trials)}; "
              f"wall p50 {statistics.median(walls) if walls else float('nan'):.0f} s; "
              f"sandbox setup mean {statistics.mean((r.get('env_setup_time') or 0) + (r.get('agent_setup_time') or 0) for r in trials):.1f} s")
    if eval_trials:
        c = collections.Counter(r.get("exit_status") for r in eval_trials)
        print(f"eval-tagged trials {len(eval_trials)}: {dict(c)}")
    for n, (t, d) in sorted(evals.items()):
        names = sorted(k[5:] for k, v in d.items() if k.startswith("eval/") and k.count("/") == 1 and isinstance(v, (int, float)))
        for name in names:
            p = f"eval/{name}"
            crit = sum(hm.num(d, f"{p}/tito_session_mismatch_rate/v1/{k}", 0) for k in CRIT)
            print(f"EVAL {name} after {n + 1} train steps ({t}): {hm.num(d, p):.3f} on {d.get(f'{p}/num_training_samples')} "
                  f"(resp len mean {hm.num(d, f'{p}/response_len/mean', 0):.0f}, total "
                  f"{hm.num(d, f'{p}/episode_total_response_length/mean', 0):.0f}; weight version {d.get(f'{p}/weight_version/mean')}; "
                  f"TITO critical mismatch {crit:.3f})")


if __name__ == "__main__":
    main()
