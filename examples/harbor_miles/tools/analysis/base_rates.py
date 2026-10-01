#!/usr/bin/env python3
"""Per-task base rates from trial logs, plus overfit-candidate selection (which tasks are learnable, i.e. mixed outcomes?).

usage: base_rates.py RUN_SPEC [RUN_SPEC ...] [--split train|eval|all (default all)] [--pick N (default 8)] [--json OUT]

Reads trials-*.jsonl (one line per finished trial, written by miles_side/hm_agent.py) of every RUN_SPEC and aggregates per
task (instance_id): attempts, successes (reward >= 1), overflow (exit status SequenceLength*/Overlong*), timeouts (Timeout* or
TimeLimitExceeded), infra failures (counted apart, not attempts), wall and decode-token medians.
Overfit candidates (ranking rule): tasks with >= 2 finished attempts and MIXED outcomes (>= 1 success and >= 1 failure),
ranked by (success rate within 25-75% first), then |success rate - 0.5|, then median decode tokens (shorter sessions first).
If there are >= N candidates, the first N are printed as an overfit task list (comma-separated, ready for a task-subset env).
Prints a markdown block; --json OUT writes {task: {n, ok, ovf, to, infra, wall_p50_s, decode_p50}}.

Ported from QWEN27B live_base_rates.py (2026-10-01): generalized to run specs; prints instead of rewriting STATUS.md;
timeouts now also count Harbor's TimeLimitExceeded (the original matched only 'Timeout', so its column was always 0).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="LOCAL_DIR | CLUSTER:RUN_NAME | CLUSTER:/abs/run/dir (one or more)")
    ap.add_argument("--split", default="all", choices=("train", "eval", "all"))
    ap.add_argument("--pick", type=int, default=8, help="size of the overfit task list (default 8)")
    ap.add_argument("--json", help="write per-task counts to this file")
    a = ap.parse_args()

    runs = []
    for s in a.runs:
        run = hm.resolve_run(s, must_exist=False)
        if run is None:
            print(f"skipping {s}: run dir not found", file=sys.stderr)
        else:
            runs.append(run)
    recs = [r for run in runs for r in hm.read_trials(run, split=a.split)]
    per = collections.defaultdict(lambda: dict(n=0, ok=0, ovf=0, to=0, infra=0, wall=[], dec=[]))
    for x in recs:
        t = per[x.get("instance_id")]
        if x.get("infra_failure"):
            t["infra"] += 1
            continue
        t["n"] += 1
        t["ok"] += int(float(x.get("reward") or 0) >= 1.0)
        es = str(x.get("exit_status") or "")
        t["ovf"] += int("SequenceLength" in es or "Overlong" in es)
        t["to"] += int("Timeout" in es or "TimeLimit" in es)
        if x.get("wall_s"):
            t["wall"].append(x["wall_s"])
        if x.get("n_output_tokens"):
            t["dec"].append(x["n_output_tokens"])  # codex trials carry no `turns` (always None)
    done = [k for k, v in per.items() if v["n"] >= 4]
    mixed = [k for k, v in per.items() if v["n"] >= 2 and 1 <= v["ok"] <= v["n"] - 1]
    dmed = lambda k: st.median(per[k]["dec"]) if per[k]["dec"] else 1e9  # noqa: E731
    cand = sorted(mixed, key=lambda k: (0 if 0.25 <= per[k]["ok"] / per[k]["n"] <= 0.75 else 1,
                                        abs(per[k]["ok"] / per[k]["n"] - 0.5), dmed(k)))
    n_all = sum(v["n"] for v in per.values())
    ok_all = sum(v["ok"] for v in per.values())
    ovf_all = sum(v["ovf"] for v in per.values())
    to_all = sum(v["to"] for v in per.values())
    walls = [w for v in per.values() for w in v["wall"]]
    lines = [f"## BASE RATES (live) -- {time.strftime('%Y-%m-%d %H:%M %Z')} -- split {a.split}: {', '.join(r.key for r in runs)}",
             f"trials {n_all} (+{sum(v['infra'] for v in per.values())} infra) over {len(per)} tasks; tasks with >=4 attempts: {len(done)}; "
             f"pass@1 {ok_all / max(1, n_all):.3f}; overflow {ovf_all} ({100 * ovf_all / max(1, n_all):.1f}%); timeouts {to_all}; "
             + (f"wall p50 {st.median(walls) / 60:.1f} min" if walls else ""),
             f"OVERFIT CANDIDATES (>=2 attempts, mixed outcomes; ranked 25-75% first, then closest to 50%, then shorter): {len(cand)} -> "
             + ", ".join(f"{k} ({per[k]['ok']}/{per[k]['n']})" for k in cand),
             "| task | attempts | successes | overflow | timeouts | wall p50 min | decode p50 (k tok) |", "|---|---|---|---|---|---|---|"]
    for k in sorted(per, key=lambda k: (-(per[k]["n"] >= 4), -per[k]["ok"], str(k))):
        v = per[k]
        lines.append(f"| {k} | {v['n']} | {v['ok']} | {v['ovf']} | {v['to']} | "
                     f"{(st.median(v['wall']) / 60 if v['wall'] else 0):.1f} | {(st.median(v['dec']) / 1e3 if v['dec'] else 0):.1f} |")
    if a.pick and len(cand) >= a.pick:
        pick = cand[:a.pick]
        lines.append("")
        lines.append("OVERFIT TASKS: " + ",".join(str(k) for k in pick) + " (base k/n: " + ", ".join(f"{per[k]['ok']}/{per[k]['n']}" for k in pick)
                     + "; median decode tok: " + ", ".join(f"{dmed(k) / 1e3:.0f}k" for k in pick)
                     + "; overflow per task: " + ", ".join(str(per[k]["ovf"]) for k in pick)
                     + "; per-session context / overflow per cap: session_lengths.py)")
    print("\n".join(lines))
    if a.json:
        out = {str(k): dict(n=v["n"], ok=v["ok"], ovf=v["ovf"], to=v["to"], infra=v["infra"],
                            wall_p50_s=st.median(v["wall"]) if v["wall"] else None,
                            decode_p50=st.median(v["dec"]) if v["dec"] else None) for k, v in per.items()}
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {a.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
