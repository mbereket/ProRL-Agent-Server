#!/usr/bin/env python3
"""Session-length distribution from Harbor trial dirs (how long are sessions, how many overflow a cap, what success is lost?).

usage: session_lengths.py RUN_SPEC [RUN_SPEC ...] [--caps 32k,48k,64k,96k,128k] [--expected N] [--cache FILE]

Reads <run>/trials/<host>/<trial>/ (full Harbor trial dirs; needs no eval dump). Per finished trial: agent/trajectory.json
steps[].metrics (prompt_tokens, completion_tokens) -> final context = last metered step's prompt + completion tokens; turns =
number of metered steps (LLM calls); result.json agent_result.n_output_tokens = decode tokens; verifier/reward.json score =
success (>= 1). Unfinished trials (any file missing / unparsable) are skipped.
Prints: final-context / decode / turns quantiles and pass@1; per cap (k = 1024 tokens): sessions whose final context exceeds
the cap and the successes among them (lost if truncated -> 0); success vs final context by quartile. Sessions are measured at
the cap the run used, so lengths above it are censored. --expected N (sessions in the full pass) adds the snapshot caveat
(unfinished sessions are the long ones -> distribution biased short). --cache FILE keeps parsed trials across invocations
(remote reads are slow); e.g. --cache ~/.cache/harbor_miles/session_lengths.json.

Ported from QWEN27B de4_session_lengths.py (2026-10-01): generalized to run specs, caps, --expected; prints instead of
rewriting STATUS.md; cache optional.
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402


def parse_caps(s: str) -> list[int]:
    return [int(float(c[:-1]) * 1024) if c.lower().endswith("k") else int(c) for c in s.split(",")]


def scan(run, cache: dict):
    """Add every finished trial of `run` to `cache` (key '<run key>|<trial dir name>')."""
    base = run.join("trials")
    for host in run.listdir(base):
        for t in run.listdir(host):
            key = f"{run.key}|{os.path.basename(t)}"
            if key in cache:
                continue
            j = lambda *p: (posixpath if run.cluster else os.path).join(t, *p)  # noqa: E731
            try:
                rew = json.loads(run.read_bytes(j("verifier", "reward.json")))
                tj = json.loads(run.read_bytes(j("agent", "trajectory.json")))
                res = json.loads(run.read_bytes(j("result.json")))
            except (FileNotFoundError, OSError, ValueError):
                continue  # trial still running / not finished
            steps = [s for s in tj.get("steps", []) if (s.get("metrics") or {}).get("prompt_tokens")]
            if not steps:
                continue
            last = steps[-1]["metrics"]
            ar = res.get("agent_result") or {}
            ex = str((res.get("exception_info") or {}).get("exception_type") or "")
            cache[key] = dict(task=os.path.basename(t).split("__")[0], final=last["prompt_tokens"] + (last.get("completion_tokens") or 0),
                              turns=len(steps), decode=ar.get("n_output_tokens") or sum(s["metrics"].get("completion_tokens") or 0 for s in steps),
                              score=float(rew.get("score", rew.get("reward", 0)) or 0), exc=ex)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="LOCAL_DIR | CLUSTER:RUN_NAME | CLUSTER:/abs/run/dir (one or more)")
    ap.add_argument("--caps", default="32k,48k,64k,96k,128k")
    ap.add_argument("--expected", type=int, help="sessions in the full pass (adds the snapshot caveat)")
    ap.add_argument("--cache", help="JSON cache of parsed trials (default: none)")
    a = ap.parse_args()
    caps = parse_caps(a.caps)

    runs = []
    for s in a.runs:
        run = hm.resolve_run(s, must_exist=False)
        if run is None:
            print(f"skipping {s}: run dir not found", file=sys.stderr)
        else:
            runs.append(run)
    cache_path = os.path.expanduser(a.cache) if a.cache else None
    cache = json.load(open(cache_path)) if cache_path and os.path.exists(cache_path) else {}
    for run in runs:
        scan(run, cache)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh)
    keys = {r.key for r in runs}
    rows = [v for k, v in cache.items() if k.rsplit("|", 1)[0] in keys]
    n = len(rows)
    if not n:
        print("no finished trials")
        return
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]  # noqa: E731
    fin = [x["final"] for x in rows]
    dec = [x["decode"] for x in rows]
    tur = [x["turns"] for x in rows]
    succ = sum(x["score"] >= 1 for x in rows)
    of = f" of {a.expected}" if a.expected else ""
    L = [f"## SESSION LENGTHS (n={n} finished sessions{of}; {time.strftime('%Y-%m-%d %H:%M %Z')}; MEASURED from Harbor trial "
         f"trajectories: {', '.join(r.key for r in runs)})",
         f"final context: p50 {q(fin, .5) / 1e3:.1f}k, p90 {q(fin, .9) / 1e3:.1f}k, max {max(fin) / 1e3:.1f}k (mean {st.mean(fin) / 1e3:.1f}k) | "
         f"decode/session: p50 {q(dec, .5) / 1e3:.1f}k, p90 {q(dec, .9) / 1e3:.1f}k | turns (LLM calls): p50 {q(tur, .5)}, p90 {q(tur, .9)}, "
         f"max {max(tur)} | pass@1 {succ / n:.3f}"]
    if a.expected:
        L.append(f"SNAPSHOT CAVEAT: {max(0, a.expected - n)} of {a.expected} sessions still running or not started at this snapshot -> the "
                 f"distribution is biased SHORT (long sessions finish last). Final numbers only from the completed pass (or "
                 f"censor-corrected with in-flight sessions as lower bounds).")
    L += ["| cap | sessions above cap | successes above cap (lost if truncated -> 0) |", "|---|---|---|"]
    for c in caps:
        over = [x for x in rows if x["final"] > c]
        lost = sum(x["score"] >= 1 for x in over)
        L.append(f"| {c // 1024}k | {len(over)} ({100 * len(over) / n:.1f}%) | {lost} ({100 * lost / max(1, succ):.1f}% of {succ} successes) |")
    srt = sorted(rows, key=lambda x: x["final"])
    k = max(1, (n + 3) // 4)
    L.append("success vs final context (quartiles): " + "; ".join(
        f"{b[0]['final'] / 1e3:.0f}-{b[-1]['final'] / 1e3:.0f}k: {sum(x['score'] >= 1 for x in b) / len(b):.2f} (n={len(b)})"
        for b in (srt[i:i + k] for i in range(0, n, k))))
    print("\n".join(L))


if __name__ == "__main__":
    main()
