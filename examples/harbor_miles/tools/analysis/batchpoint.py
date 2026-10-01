#!/usr/bin/env python3
"""Per-step systems record of a harbor_miles run (batch-point validation: are engines, sandboxes and nodes saturated?).

usage: batchpoint.py --rbs N JOBLOG [JOBLOG...] [--nodes NODE_CSV ...] [--trials TRIALS_JSONL ...]   (local files)
       batchpoint.py --run RUN_SPEC [--rbs N] [--jobs JOBID ...]
         (job logs, node-*.csv and trials-*.jsonl of the run; --rbs defaults to --rollout-batch-size from args-*.txt)

Per step N (window = previous step's end .. this step's end, by the `perf N:` timestamps):
  wall s, trainer wait for data s (perf/train_wait_time), train s (perf/actor_train_time), weight sync s
  (perf/update_weights_time), groups kept/dropped (zero-variance %), async queue / stale-filtered groups,
  per-engine SGLang decode stats in the window (running requests mean/max, KV 'full token usage' mean/max, queued requests
  max), sandboxes in flight and job CPU cores / memory per node (node monitor CSVs), trials ended in the window.

Ported from PATH-B bin/batchpoint.py (2026-10-01): same CLI and output; adds --run RUN_SPEC (local or remote) and --jobs.
"""
from __future__ import annotations

import ast
import collections
import csv
import io
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402


def g(d, *keys):
    for k in keys:
        for kk, v in d.items():
            if kk.endswith(k):
                return v
    return None


def fmt(v, f="{:.0f}"):
    return f.format(v) if isinstance(v, (int, float)) else "-"


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    logs, nodes, trials_paths, jobs, rbs, spec = [], [], [], [], None, None
    cur = logs
    it = iter(args)
    for a in it:
        if a == "--nodes": cur = nodes; continue  # noqa: E701,E702
        if a == "--trials": cur = trials_paths; continue  # noqa: E701,E702
        if a == "--jobs": cur = jobs; continue  # noqa: E701,E702
        if a == "--rbs": rbs = int(next(it)); continue  # noqa: E701,E702
        if a == "--run": spec = next(it); continue  # noqa: E701,E702
        cur.append(a)

    # inputs as (name, text)
    log_texts, node_texts, trial_texts = [], [], []
    if spec:
        run = hm.resolve_run(spec)
        log_texts = [("joblogs", hm.read_joblogs(run, jobids=jobs or None, local_paths=logs or None))]
        node_texts = [(os.path.basename(p), run.read_text(p)) for p in run.files(r"node-(\d+)-.*\.csv")
                      if not jobs or os.path.basename(p).split("-")[1] in jobs]
        trial_texts = [(os.path.basename(p), run.read_text(p)) for p in run.files(r"trials-.*\.jsonl")
                       if not jobs or os.path.basename(p)[7:-6] in jobs]
        if rbs is None:
            v = hm.arg_value(hm.read_args(run), "--rollout-batch-size")
            rbs = int(v) if v and v.isdigit() else None
    else:
        log_texts = [(p, hm.ANSI_RE.sub("", open(p, errors="replace").read())) for p in logs]
    node_texts += [(p, open(p).read()) for p in nodes]
    trial_texts += [(p, open(p).read()) for p in trials_paths]

    perf, perf_t = collections.defaultdict(dict), {}
    eng = collections.defaultdict(list)  # engine key -> [(t, running, usage, queue)]
    dec = re.compile(r"\((?:CommandActor|SGLang\w*) pid=(\d+)(?:, ip=([\d.]+))?\).*\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0\] Decode batch, "
                     r"#running-req: (\d+), #full token: \d+, full token usage: ([\d.]+).*#queue-req: (\d+)")
    for _name, text in log_texts:
        for line in text.splitlines():
            m = re.search(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].* - perf (\d+): (\{.*\})\s*$", line)
            if m:
                try:
                    d = ast.literal_eval(m.group(3))
                except Exception:  # noqa: BLE001
                    continue
                n = int(m.group(2))
                perf[n].update(d)
                t = hm.parse_ts(m.group(1))
                perf_t[n] = max(perf_t.get(n, 0), t)
                continue
            m = dec.search(line)
            if m:
                pid, ip, t, run_, use, q = m.groups()
                eng[f"{ip or 'head'}:{pid}"].append((hm.parse_ts(t), int(run_), float(use), int(q)))

    node_rows = collections.defaultdict(list)
    for p, text in node_texts:
        host = re.sub(r".*node-\d+-|\.csv$", "", p)
        for r in csv.DictReader(io.StringIO(text)):
            try:
                node_rows[host].append((int(r["ts"]), float(r["job_cpu_cores"] or "nan"), float(r["job_mem_gb"] or "nan"),
                                        int(r["sandboxes"] or 0)))
            except (ValueError, KeyError, TypeError):
                pass
    trials = []
    for _p, text in trial_texts:
        for line in text.splitlines():
            if line.strip().startswith("{"):
                try:
                    trials.append(json.loads(line))
                except ValueError:
                    pass

    steps = sorted(perf_t)
    print("| step | end | wall s | wait-for-data s | train s | wsync s | groups kept/dropped (drop%) | queue / stale-filtered |"
          " engines: running mean/max | KV usage mean/max | engine queue max | sandboxes per node (mean) | job CPU cores per node (mean/max) |"
          " job mem GB per node (max) | trials ended (overlong) |")
    print("|" + "---|" * 15)
    prev = None
    for n in steps:
        d, t1 = perf[n], perf_t[n]
        t0 = perf_t.get(prev, t1 - float(g(d, "perf/step_time") or 0)) if prev is not None else None
        drops = sum(v for k, v in d.items() if "dynamic_filter/drop_zero_std" in k and isinstance(v, (int, float)))
        lo = t0 if t0 is not None else t1 - 3600
        er = [(r, u, q) for e in eng.values() for (t, r, u, q) in e if lo < t <= t1]
        per_eng_run = [statistics.mean([r for (t, r, u, q) in e if lo < t <= t1]) for e in eng.values()
                       if any(lo < t <= t1 for (t, *_x) in e)]
        nd = {h: [(c, m, s) for (t, c, m, s) in rows if lo < t <= t1] for h, rows in node_rows.items()}
        sb = " / ".join(fmt(statistics.mean([s for c, m, s in v])) for h, v in sorted(nd.items()) if v) or "-"
        cpu = " / ".join(f"{statistics.mean([c for c, m, s in v]):.0f}/{max(c for c, m, s in v):.0f}" for h, v in sorted(nd.items()) if v) or "-"
        mem = " / ".join(fmt(max(m for c, m, s in v)) for h, v in sorted(nd.items()) if v) or "-"
        tr = [r for r in trials if lo < (r.get("t_end") or 0) <= t1]
        ovl = sum(hm.is_overlong(r) for r in tr)
        run_s = (f"{statistics.mean(per_eng_run):.1f}/{max(r for r, u, q in er)}" if er else "-")
        use_s = (f"{statistics.mean([u for r, u, q in er]):.2f}/{max(u for r, u, q in er):.2f}" if er else "-")
        print(f"| {n} | {hm.fmt_ts(t1, '%H:%M:%S')} | {fmt(t1 - t0) if t0 else '-'} |"
              f" {fmt(g(d, 'perf/train_wait_time'))} | {fmt(g(d, 'perf/actor_train_time'))} | {fmt(g(d, 'perf/update_weights_time'), '{:.1f}')} |"
              f" {fmt(rbs)}/{fmt(drops)} ({fmt(100 * drops / (drops + rbs) if rbs else None)}%) |"
              f" {fmt(g(d, 'rollout/fully_async/queue_size'))} / {fmt(g(d, 'rollout/fully_async/stale_groups_filtered'))} |"
              f" {run_s} (n={len(per_eng_run)}) | {use_s} | {max((q for r, u, q in er), default='-')} | {sb} | {cpu} | {mem} |"
              f" {len(tr)} ({ovl}) |")
        prev = n


if __name__ == "__main__":
    main()
