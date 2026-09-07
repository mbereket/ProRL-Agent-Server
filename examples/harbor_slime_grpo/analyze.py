#!/usr/bin/env python3
"""Assess a run from its run directory (or a fetched copy of it).

    python analyze.py <run_dir> [--caps 32768,65536,131072] [--workers N] [--step-sessions N]
                      [--no-transcripts] [--max-chars 1500]

Reads rollout_results/task_*/ses_*.json and train.jsonl, writes <run_dir>/analysis/:
  sessions.csv        one row per session: task, status, reward, turns, traces, tokens, timing, node
  summary.txt         what is printed: outcomes, reward histogram, per-step and per-task tables,
                      chain breaks (sessions split into several traces, with the cause), trace
                      lengths against caps, wallclock distribution, throughput and step-time estimate
  transcripts/<task>/<session>.md   readable trajectories (prompt, every turn, tool calls and
                      results truncated to --max-chars, verifier output tail)

Trace length = prompt + response tokens of the longest trace in a session (what the trainer
must fit). Wallclock per session comes from Polar's timing (queue, init, run, postrun).
Throughput: observed sessions/hour over the batch, and a greedy simulation of the observed
durations on --workers concurrent sandbox slots (default: topology max_run_workers x gateway
nodes) for a step of --step-sessions sessions (default: this batch).
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, round(q * (len(xs) - 1))))]


def load_sessions(run_dir: Path) -> list[dict]:
    prompts = []
    train = run_dir / "train.jsonl"
    if train.is_file():
        for line in train.open():
            md = json.loads(line)["metadata"]
            prompts.append(md.get("source_id") or md.get("instance_id") or md.get("task_dir") or "")
    rows = []
    for p in sorted((run_dir / "rollout_results").glob("*/ses_*.json")):
        d = json.loads(p.read_text())
        tr = d.get("trajectory") or {}
        md = tr.get("metadata") or {}
        ev = md.get("evaluation") or {}
        traces = tr.get("traces") or []
        lengths = [len(t.get("prompt_ids", [])) + len(t.get("response_ids", [])) for t in traces]
        timing = d.get("timing") or {}
        gi = (d.get("metadata") or {}).get("group_id", md.get("group_id"))
        step = (d.get("metadata") or {}).get("rollout_step", md.get("rollout_step"))
        rows.append({
            "step": step,
            "task": prompts[gi] if isinstance(gi, int) and gi < len(prompts) else f"group{gi}",
            "group": gi,
            "session_id": d["session_id"],
            "status": d.get("status"),
            "error": (d.get("error") or tr.get("error") or "").replace("\n", " ")[:120],
            "reward": float(ev.get("reward") or 0.0),
            "evaluated": int(bool(ev)),
            "verifier_exit": ev.get("verifier_exit_code"),
            "turns": md.get("record_count", 0),
            "traces": len(traces),
            "longest_trace_tokens": max(lengths) if lengths else 0,
            "response_tokens": sum(len(t.get("response_ids", [])) for t in traces),
            "finish": traces[-1].get("finish_reason") if traces else "",
            "queue_s": timing.get("register_to_init_queue_ms", 0) / 1000,
            "init_s": timing.get("init_ms", 0) / 1000,
            "run_s": timing.get("run_ms", 0) / 1000,
            "postrun_s": timing.get("postrun_ms", 0) / 1000,
            "node": d.get("node_id", ""),
            "chain_break": first_chain_break(traces),
            "_path": p,
            "_traces": traces,
            "_verifier_tail": ev.get("verifier_output_tail", ""),
        })
    return rows


def first_chain_break(traces: list[dict]) -> str:
    """Why a session became more than one trace: the first replayed message that is not
    what the model generated (role and a snippet), or the first message the next prompt
    added beyond the previous trace's history."""
    if len(traces) < 2:
        return ""
    a, b = traces[0], traces[1]
    expected = a.get("prompt_messages", []) + a.get("response_messages", [])
    got = b.get("prompt_messages", [])

    def brief(m: dict) -> str:
        c = m.get("content")
        text = c if isinstance(c, str) else json.dumps(c) if c else ""
        calls = ",".join(t["function"]["name"] for t in m.get("tool_calls") or [])
        return f"{m.get('role')}{'(' + calls + ')' if calls else ''}: {text[:60]!r}"

    for x, y in zip(expected, got):
        if brief(x) != brief(y):
            return f"replayed {brief(y)} where generated {brief(x)}"
    added = got[len(expected):]
    if added:
        return f"next prompt added {brief(added[0])}"
    return "token divergence inside an identical message list (template rendering)"


def simulate(durations: list[float], workers: int) -> float:
    """Wallclock of running these durations on `workers` slots, longest first (greedy)."""
    slots = [0.0] * max(1, workers)
    for d in sorted(durations, reverse=True):
        i = slots.index(min(slots))
        slots[i] += d
    return max(slots)


def summarize(rows: list[dict], caps: list[int], workers: int, step_sessions: int) -> str:
    out = []
    n = len(rows)
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    fin = [r for r in rows if r["evaluated"]]
    out.append(f"sessions: {n}  status: {by_status}  evaluated (verifier ran): {len(fin)}")
    if fin:
        rewards = [r["reward"] for r in fin]
        out.append(f"reward: mean {statistics.fmean(rewards):.3f}  success (>0) {sum(r > 0 for r in rewards)}/{len(fin)}"
                   f"  full (>=1) {sum(r >= 1 for r in rewards)}/{len(fin)}")
        bins = [0, 0.001, 0.25, 0.5, 0.75, 1.0]
        labels = ["=0", "(0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "=1"]
        hist = [0] * 6
        for r in rewards:
            hist[5 if r >= 1 else next(i for i in range(5) if r < bins[i + 1])] += 1
        out.append("reward histogram: " + "  ".join(f"{l} {c}" for l, c in zip(labels, hist)))
    if rows:
        L = [r["longest_trace_tokens"] for r in rows if r["traces"]]
        out.append(f"longest-trace tokens: p50 {pct(L, .5):.0f}  p90 {pct(L, .9):.0f}  p95 {pct(L, .95):.0f}  max {max(L) if L else 0}")
        for c in caps:
            over = sum(1 for x in L if x > c)
            out.append(f"  over {c}: {over}/{len(L)} = {over / max(1, len(L)):.2f}")
        T = [r["turns"] for r in rows]
        out.append(f"turns: p50 {pct(T, .5):.0f}  p90 {pct(T, .9):.0f}  max {max(T)}   traces/session: "
                   f"mean {statistics.fmean(r['traces'] for r in rows):.2f}  max {max(r['traces'] for r in rows)}"
                   f"   finish: {dict(sorted(((f, sum(1 for r in rows if r['finish'] == f)) for f in {r['finish'] for r in rows})))}")
        run = [r["run_s"] for r in rows]
        out.append(f"agent wallclock (s): p50 {pct(run, .5):.0f}  p90 {pct(run, .9):.0f}  max {max(run):.0f}"
                   f"   init p50 {pct([r['init_s'] for r in rows], .5):.0f}  postrun p50 {pct([r['postrun_s'] for r in rows], .5):.0f}"
                   f"  queue max {max(r['queue_s'] for r in rows):.0f}")
        tps = [r["response_tokens"] / r["run_s"] for r in rows if r["run_s"] > 0 and r["response_tokens"]]
        if tps:
            out.append(f"response tokens/s per session: p50 {pct(tps, .5):.1f}  p90 {pct(tps, .9):.1f}")
        wall = max(r["queue_s"] + r["init_s"] + r["run_s"] + r["postrun_s"] for r in rows)
        total = [r["init_s"] + r["run_s"] + r["postrun_s"] for r in rows]
        out.append(f"observed batch wallclock: {wall / 60:.1f} min for {n} sessions  ({n / (wall / 3600):.1f} sessions/h)")
        sim = simulate(total, workers)
        out.append(f"estimate on {workers} slots: this batch {sim / 60:.1f} min; a step of {step_sessions} sessions "
                   f"{simulate((total * (step_sessions // max(1, n) + 1))[:step_sessions], workers) / 60:.1f} min")
    steps = sorted({r["step"] for r in rows if r["step"] is not None})
    if len(steps) > 1:
        out.append("")
        out.append(f"{'step':>4} {'sess':>4} {'succ':>5} {'mean_r':>6} {'traces/s':>8} {'p50_tok':>8} {'over_cap':>8} {'p50_run_s':>9} {'max_run_s':>9}")
        for st in steps:
            rs = [r for r in rows if r["step"] == st]
            L_ = [r["longest_trace_tokens"] for r in rs]
            out.append(f"{st:>4} {len(rs):>4} {sum(1 for r in rs if r['reward'] > 0) / len(rs):>5.2f} "
                       f"{statistics.fmean(r['reward'] for r in rs):>6.2f} {statistics.fmean(r['traces'] for r in rs):>8.2f} "
                       f"{pct(L_, .5):>8.0f} {sum(1 for x in L_ if x > caps[-1]) / len(rs):>8.2f} "
                       f"{pct([r['run_s'] for r in rs], .5):>9.0f} {max(r['run_s'] for r in rs):>9.0f}")
    breaks = [r for r in rows if r["traces"] > 1]
    if breaks:
        out.append("")
        out.append(f"chain breaks: {len(breaks)} session(s) with >1 trace (should be 0 for codex)")
        for r in sorted(breaks, key=lambda r: -r["traces"])[:10]:
            out.append(f"  {r['task'][:24]} step {r['step']} {r['traces']} traces: {r['chain_break']}")
    # per task
    tasks = sorted({r["task"] for r in rows})
    out.append("")
    out.append(f"{'task':<40} {'n':>3} {'succ':>4} {'mean_r':>6} {'p50_tok':>8} {'max_tok':>8} {'p50_run_s':>9} {'turns':>5}")
    for t in tasks:
        rs = [r for r in rows if r["task"] == t]
        out.append(f"{t[:40]:<40} {len(rs):>3} {sum(1 for r in rs if r['reward'] > 0):>4} "
                   f"{statistics.fmean(r['reward'] for r in rs):>6.2f} {pct([r['longest_trace_tokens'] for r in rs], .5):>8.0f} "
                   f"{max(r['longest_trace_tokens'] for r in rs):>8} {pct([r['run_s'] for r in rs], .5):>9.0f} "
                   f"{pct([r['turns'] for r in rs], .5):>5.0f}")
    errs = [r for r in rows if r["error"]]
    if errs:
        out.append("")
        out.append(f"errors ({len(errs)}):")
        for r in errs[:20]:
            out.append(f"  {r['task'][:30]} {r['session_id'][-8:]} {r['status']}: {r['error']}")
    return "\n".join(out)


def transcript(r: dict, max_chars: int) -> str:
    def clip(s: object) -> str:
        s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
        return s if len(s) <= max_chars else s[:max_chars] + f"\n... [{len(s) - max_chars} more chars]"

    lines = [f"# {r['task']}  session {r['session_id']}",
             f"status {r['status']}  reward {r['reward']}  turns {r['turns']}  traces {r['traces']}  "
             f"longest trace {r['longest_trace_tokens']} tok  run {r['run_s']:.0f}s  finish {r['finish']}  node {r['node']}", ""]
    for ti, t in enumerate(r["_traces"]):
        if len(r["_traces"]) > 1:
            lines.append(f"## trace {ti + 1}/{len(r['_traces'])}  ({len(t.get('prompt_ids', []))} prompt + {len(t.get('response_ids', []))} response tokens)")
        for m in t.get("prompt_messages", []):
            lines += [f"### {m.get('role')}", clip(m.get("content") or ""), ""]
        for m in t.get("response_messages", []):
            role = m.get("role")
            if role == "assistant":
                if m.get("content"):
                    lines += ["### assistant", clip(m["content"]), ""]
                for call in m.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    lines += [f"### assistant -> tool `{fn.get('name')}`", "```", clip(fn.get("arguments") or ""), "```", ""]
            elif role == "tool":
                lines += ["### tool result", "```", clip(m.get("content") or ""), "```", ""]
            else:
                lines += [f"### {role}", clip(m.get("content") or ""), ""]
    if r["_verifier_tail"]:
        lines += ["## verifier output (tail)", "```", r["_verifier_tail"][-max_chars:], "```"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--caps", default="32768,65536,131072")
    p.add_argument("--workers", type=int, help="concurrent sandbox slots for the estimate (default from topology.yaml)")
    p.add_argument("--step-sessions", type=int, help="sessions per training step to estimate (default: this batch)")
    p.add_argument("--no-transcripts", action="store_true")
    p.add_argument("--max-chars", type=int, default=1500)
    a = p.parse_args()

    run_dir = Path(a.run_dir)
    rows = load_sessions(run_dir)
    if not rows:
        sys.exit(f"no rollout_results/*/ses_*.json under {run_dir}")
    workers = a.workers
    if not workers:
        try:
            import yaml
            topo = yaml.safe_load((run_dir / "topology.yaml").read_text())
            nodes = topo["gateway"]["nodes"]
            workers = sum(int(n.get("max_run_workers", 16)) for n in nodes)
        except Exception:
            workers = 16
    caps = [int(c) for c in a.caps.split(",") if c]
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)

    fields = [k for k in rows[0] if not k.startswith("_")]
    with open(out / "sessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    summary = summarize(rows, caps, workers, a.step_sessions or len(rows))
    (out / "summary.txt").write_text(summary + "\n")
    print(summary)

    if not a.no_transcripts:
        for r in rows:
            d = out / "transcripts" / r["task"][:60].replace("/", "_")
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{r['session_id']}.md").write_text(transcript(r, a.max_chars))
        print(f"\ntranscripts: {out / 'transcripts'}  ({len(rows)} files)")
    print(f"sessions.csv, summary.txt: {out}")


if __name__ == "__main__":
    main()
