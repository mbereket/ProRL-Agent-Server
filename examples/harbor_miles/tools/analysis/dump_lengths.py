#!/usr/bin/env python3
"""Per-session lengths, overflow at candidate caps and success vs length from Miles eval dumps (needs torch).

usage: dump_lengths.py eval_A.pt [eval_B.pt ...] [--caps 32k,48k,64k,96k,128k] [--rollout-cap 128k] [--json OUT]

Input: local copies of <run>/dumps/eval_0.pt (Miles --save-debug-rollout-data, evaluation=True -> stem eval_0), e.g. one per
half of a base pass. Each sample: tokens (prompt + all turns, token-in-token-out), response_length, loss_mask (1 = sampled),
reward, metadata (instance_id, exit_status). Final context = len(tokens). A session "overflows cap C" if its natural final
context (at the run's cap, --rollout-cap) exceeds C; sessions that hit the rollout cap (exit SequenceLengthLimitExceeded or
within 64 tokens of it) are censored there. --json OUT: rows, the cap table and per-task counts (n, ok, ctx_med and
over<cap k> per cap; at the rollout cap it counts sessions that hit it).

Ported from QWEN27B de4_dump_analysis.py (2026-10-01): logic and output unchanged; caps and rollout cap configurable
(were 32k-128k and 128k).
"""
from __future__ import annotations

import json
import statistics as st
import sys


def parse_caps(s: str) -> list[int]:
    return [int(float(c[:-1]) * 1024) if c.lower().endswith("k") else int(c) for c in s.split(",")]


def load(paths):
    import torch  # lazy: only this tool needs torch
    rows = []
    for p in paths:
        d = torch.load(p, weights_only=False)
        for s in d["samples"]:
            md = s.get("metadata") or {}
            r = s.get("reward")
            if isinstance(r, dict):
                r = r.get("score", next(iter(r.values()), 0.0))
            rows.append(dict(task=md.get("instance_id") or md.get("task_id") or str(s.get("group_index")),
                             total=len(s.get("tokens") or []), sampled=sum(s.get("loss_mask") or []),
                             resp=s.get("response_length") or 0, reward=float(r or 0.0),
                             exit=str(md.get("exit_status") or ""), status=str(s.get("status") or "")))
    return rows


def opt(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    out = opt(argv, "--json")
    caps = parse_caps(opt(argv, "--caps", "32k,48k,64k,96k,128k"))
    cap = parse_caps(opt(argv, "--rollout-cap", "128k"))[0]
    skip = {i + 1 for i, a in enumerate(argv) if a in ("--json", "--caps", "--rollout-cap")}
    paths = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in skip]
    rows = load(paths)
    n = len(rows)
    if not n:
        print("no samples")
        return
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]  # noqa: E731
    hit = lambda x: "SequenceLength" in x["exit"] or x["total"] >= cap - 64  # noqa: E731
    tot = [x["total"] for x in rows]
    smp = [x["sampled"] for x in rows]
    hitc = sum(1 for x in rows if hit(x))
    print(f"sessions {n}; final ctx mean {st.mean(tot) / 1e3:.1f}k p50 {q(tot, .5) / 1e3:.1f}k p90 {q(tot, .9) / 1e3:.1f}k max {max(tot) / 1e3:.1f}k; "
          f"sampled/session mean {st.mean(smp) / 1e3:.1f}k; hit {cap // 1024}k: {hitc} ({100 * hitc / n:.1f}%)")
    succ = [x for x in rows if x["reward"] >= 1.0]
    print(f"pass@1 {len(succ) / n:.3f}; successes {len(succ)}")
    print("| cap | sessions over cap (truncated) | successes over cap (lost if truncated) | pass@1 if truncation -> 0 |")
    print("|---|---|---|---|")
    table = []
    for c in caps:
        over = [x for x in rows if x["total"] > c]
        lost = [x for x in over if x["reward"] >= 1.0]
        p = (len(succ) - len(lost)) / n
        table.append(dict(cap=c, over=len(over), over_frac=len(over) / n, lost_succ=len(lost), pass1_trunc=p))
        print(f"| {c // 1024}k | {len(over)} ({100 * len(over) / n:.1f}%) | {len(lost)} ({100 * len(lost) / max(1, len(succ)):.1f}% of successes) | {p:.3f} |")
    # success vs length (quintiles of final context)
    srt = sorted(rows, key=lambda x: x["total"])
    k = max(1, n // 5)
    print("success vs final context (quintiles):")
    for i in range(0, n, k):
        b = srt[i:i + k]
        print(f"  ctx {b[0]['total'] / 1e3:.0f}k-{b[-1]['total'] / 1e3:.0f}k: n {len(b)}, pass {sum(x['reward'] >= 1 for x in b) / len(b):.2f}")
    per = {}
    for x in rows:
        t = per.setdefault(x["task"], dict(n=0, ok=0, ctx=[], **{f"over{c // 1024}": 0 for c in caps}))
        t["n"] += 1
        t["ok"] += int(x["reward"] >= 1)
        t["ctx"].append(x["total"])
        for c in caps:
            t[f"over{c // 1024}"] += int(hit(x) if c >= cap else x["total"] > c)
    for t in per.values():
        t["ctx_med"] = st.median(t["ctx"])
        del t["ctx"]
    if out:
        with open(out, "w") as fh:
            json.dump(dict(rows=rows, caps=table, per_task=per), fh, indent=1)
        print("wrote", out)


if __name__ == "__main__":
    main()
