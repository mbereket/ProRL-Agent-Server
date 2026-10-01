#!/usr/bin/env python3
"""Watch a running harbor_miles job: job state, launch args, per-step lines, trials per agent server, first error.

usage: watch_run.py CLUSTER JOBID RUN_NAME_OR_ABS_DIR [--local-mirror DIR] [--interval S] [--alert-wall S] [--marker REGEX]

Every --interval seconds (default 180): `scjob.py CLUSTER status JOBID` (prints the state when it changes); while the job
is RUNNING / COMPLETING / gone, mirrors trials-<job>.jsonl, agent_servers-<job>.txt, args-<job>.txt from the run dir and
<job>-*.log from the job-log dir (scrsync.py) into --local-mirror (default ./runs-mirror/<cluster>/<run>), then prints:
  args: the layout / sequence / lr / in-flight flags from args-<job>.txt (once)
  check1: finished trials per agent server (when the number of servers changes)
  env: job-log lines matching --marker (default '\\[diag-cfg\\]'), once each
  step N @HH:MM: wall, train, wait, wsync, train/rollout KL, ppo_kl, grad norm, reward, unfiltered reward, dropped groups,
    staleness, nan-logprob samples (ALERT if wall > --alert-wall, default 1200 s)
  ERROR: the first OOM / hm_die / NCCL / weight-update failure / Traceback line (once)
Exits when the job is gone from squeue. The run dir and job-log dir are resolved as in hmruns.py (shared miles/runs and
miles/joblogs first, then legacy roots) and re-resolved each round until they exist (a pending job has no log yet).

Ported from DIAG bin/watch_27b.py (2026-10-01): generalized to any cluster / job / run (was dfw, path-b and one run);
same checks and output lines.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hmruns as hm  # noqa: E402

CT = hm.CLUSTER_TOOLS
PY = hm.CT_PYTHON
ERR_RE = re.compile(r".*(CUDA out of memory|OutOfMemoryError|hm_die|not ready after|cannot stat|RuntimeError: POST|"
                    r"update_weights.*(fail|error)|NCCL.*(error|timeout)|Traceback \(most recent call last\):\n(?!.*freeze_gc)).*")


def sh(*cmd: str) -> str:
    return subprocess.run([PY, *cmd], capture_output=True, text=True, cwd=CT).stdout


def find_joblog_dir(run: hm.Run, job: str) -> str | None:
    for d in run.joblog_dirs():
        if any(os.path.basename(p).startswith(f"{job}-") for p in run.listdir(d)):
            return d
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cluster")
    ap.add_argument("jobid")
    ap.add_argument("run", help="RUN_NAME or absolute remote run dir")
    ap.add_argument("--local-mirror", help="local mirror dir (default ./runs-mirror/<cluster>/<run>)")
    ap.add_argument("--interval", type=float, default=180)
    ap.add_argument("--alert-wall", type=float, default=1200, help="alert if a step's wall time exceeds this (s)")
    ap.add_argument("--marker", default=r"\[diag-cfg\]", help="regex of job-log lines to echo once (env / config dumps)")
    a = ap.parse_args()
    cl, J = a.cluster, a.jobid
    if not J.isdigit():
        sys.exit("JOBID must be numeric")
    hm.remote_fs(cl)  # re-executes under the cluster-tools venv before any output if needed
    L = os.path.abspath(a.local_mirror or os.path.join("runs-mirror", cl, os.path.basename(a.run.rstrip("/"))))
    os.makedirs(f"{L}/joblogs", exist_ok=True)
    marker = re.compile(a.marker + r"[^\n]*")
    print(f"watching {cl} job {J} run {a.run}; mirror {L}", flush=True)

    run = logdir = None
    args_done = False
    last_st = None
    servers_seen = set()
    reported_servers = 0
    seen_steps = set()
    err = False
    step_t = {}
    while True:
        out = sh("scjob.py", cl, "status", J)
        if "JOBID" in out:
            st = next((ln.split()[1] for ln in out.splitlines() if ln.split()[:1] == [J]), "GONE")
            if st != last_st:
                print(f"job {J}: {st} ({datetime.datetime.now():%H:%M})", flush=True)
                last_st = st
            if st in ("RUNNING", "GONE", "COMPLETING"):
                if run is None:
                    run = hm.resolve_run(f"{cl}:{a.run}", must_exist=False)
                    if run:
                        print(f"run dir: {run.path}", flush=True)
                if run is not None and logdir is None:
                    logdir = find_joblog_dir(run, J)
                    if logdir:
                        print(f"job-log dir: {logdir}", flush=True)
                if run is not None:
                    sh("scrsync.py", cl, f"{run.path}/", f"{L}/", f"--include=trials-{J}.jsonl", f"--include=agent_servers-{J}.txt",
                       f"--include=args-{J}.txt", "--exclude=*")
                if logdir:
                    sh("scrsync.py", cl, f"{logdir}/", f"{L}/joblogs/", f"--include={J}-*.log", "--exclude=*")
                af = f"{L}/args-{J}.txt"
                if not args_done and os.path.exists(af):
                    tok = open(af).read().split()
                    args_done = True
                    av = lambda k: hm.arg_value(tok, k, "-")  # noqa: E731
                    print("args: actor nodes x gpus", av("--actor-num-nodes"), "x", av("--actor-num-gpus-per-node"), "| rollout gpus",
                          av("--rollout-num-gpus"), "| engine TP", av("--rollout-num-gpus-per-engine"), "| TP", av("--tensor-model-parallel-size"),
                          "CP", av("--context-parallel-size"), "| linear-cp-mode", av("--linear-cp-mode"), "| max tok/GPU", av("--max-tokens-per-gpu"),
                          "| max-seq-len", av("--max-seq-len"), "| lr", av("--lr"), "| in-flight", av("--async-max-concurrent-samples"),
                          "| logprob chunk", av("--log-probs-chunk-size"), flush=True)
                # check 1: trials per agent server (node)
                tf = f"{L}/trials-{J}.jsonl"
                if os.path.exists(tf):
                    rows = []
                    for x in open(tf):
                        try:
                            rows.append(json.loads(x))
                        except ValueError:
                            pass
                    c = collections.Counter(r.get("agent_server") for r in rows)
                    if len(c) != reported_servers:
                        reported_servers = len(c)
                        print(f"check1: finished trials by agent server: {dict(c)}", flush=True)
                for lg in sorted(glob.glob(f"{L}/joblogs/{J}-*.log")):
                    txt = hm.ANSI_RE.sub("", open(lg, errors="replace").read())
                    for m in marker.finditer(txt):
                        if m.group(0) not in servers_seen:
                            servers_seen.add(m.group(0))
                            print("env: " + m.group(0)[:200], flush=True)
                    for m in re.finditer(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].* - step (\d+): (\{.*\})", txt):
                        n = int(m.group(2))
                        if n in seen_steps or "log_utils" not in m.group(0):
                            continue
                        try:
                            d = hm.literal_dict(m.group(3))
                        except (ValueError, SyntaxError):
                            continue
                        seen_steps.add(n)
                        t = hm.parse_ts(m.group(1))
                        step_t[n] = t
                        drops = sum(v for k, v in d.items() if k.startswith("rollout/dynamic_filter/drop_") and isinstance(v, (int, float)))
                        wall = (t - step_t[n - 1]) if n - 1 in step_t else None
                        g = lambda k, f="{:.3g}": f.format(d[k]) if isinstance(d.get(k), (int, float)) else "-"  # noqa: E731
                        print(f"step {n} @{hm.fmt_ts(t)}: wall {wall and round(wall)}s train {g('perf/actor_train_time', '{:.0f}')}s "
                              f"wait {g('perf/train_wait_time', '{:.0f}')}s wsync {g('perf/update_weights_time', '{:.1f}')}s "
                              f"tr_ro_kl {g('train/train_rollout_kl')} ppo_kl {g('train/ppo_kl')} grad {g('train/grad_norm')} "
                              f"reward {g('rollout/raw_reward')} unfilt {g('rollout/raw_reward_unfiltered')} dropped_groups {drops:.0f} "
                              f"stale {g('rollout/fully_async/avg_staleness')} nan {g('harbor/nan_logprob_samples')}", flush=True)
                        if wall and wall > a.alert_wall:
                            print(f"ALERT step {n} wall {wall:.0f}s > {a.alert_wall / 60:.0f} min", flush=True)
                    if not err:
                        m = ERR_RE.search(txt)
                        if m and "freeze_gc" not in txt[max(0, m.start() - 50): m.end() + 800]:
                            print("ERROR: " + m.group(0)[:220], flush=True)
                            err = True
            if st == "GONE":
                print("job gone", flush=True)
                break
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
