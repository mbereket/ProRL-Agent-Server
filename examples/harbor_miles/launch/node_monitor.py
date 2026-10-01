#!/usr/bin/env python3
"""Per-node resource sampler for sandbox-density measurements (started by node_entry.sh on every node).

    node_monitor.py OUT_CSV AGENT_SERVER_URL [INTERVAL_S]

Every INTERVAL_S (default 30) appends one row:
  ts, job_cpu_cores (this job's cgroup CPU usage over the interval, in cores), job_cpu_limit_cores,
  job_mem_gb (cgroup memory.current), job_mem_limit_gb, node_mem_avail_gb, load1, ncpu_online,
  sandboxes (agent server /stats: in-flight trials), max_sandboxes
Cgroup v2 and v1 (cpuacct/memory) are both read; missing values are left empty. Never raises.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request


def _cgroup_paths() -> dict[str, str]:
    paths: dict[str, str] = {}
    try:
        for line in open("/proc/self/cgroup"):
            _, ctrls, path = line.rstrip("\n").split(":", 2)
            if ctrls == "":
                paths["v2"] = "/sys/fs/cgroup" + path
            for c in ctrls.split(","):
                if c in ("cpuacct", "memory", "cpu"):
                    paths[c] = f"/sys/fs/cgroup/{ctrls}{path}"
    except OSError:
        pass
    return paths


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _job_cgroup(paths: dict[str, str], key: str, leaf: str) -> str | None:
    """Walk up from this process's cgroup to the slurm job level (job_<id>), where the limits live."""
    base = paths.get(key)
    if not base:
        return None
    job = os.environ.get("SLURM_JOB_ID", "")
    cur = base
    while cur and cur != "/sys/fs/cgroup":
        if os.path.basename(cur) == f"job_{job}" and os.path.exists(os.path.join(cur, leaf)):
            return cur
        cur = os.path.dirname(cur)
    return base if os.path.exists(os.path.join(base, leaf)) else None


def cpu_usage_s(paths: dict[str, str]) -> float | None:
    cg = _job_cgroup(paths, "v2", "cpu.stat")
    if cg:
        for line in (_read(os.path.join(cg, "cpu.stat")) or "").splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1]) / 1e6
    cg = _job_cgroup(paths, "cpuacct", "cpuacct.usage")
    v = _read(os.path.join(cg, "cpuacct.usage")) if cg else None
    return int(v) / 1e9 if v else None


def cpu_limit_cores(paths: dict[str, str]) -> float | None:
    cg = _job_cgroup(paths, "v2", "cpuset.cpus.effective")
    spec = _read(os.path.join(cg, "cpuset.cpus.effective")) if cg else None
    if not spec:
        try:
            return float(len(os.sched_getaffinity(0)))
        except Exception:
            return None
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return float(n)


def mem(paths: dict[str, str]) -> tuple[float | None, float | None]:
    cg = _job_cgroup(paths, "v2", "memory.current")
    if cg:
        cur, lim = _read(os.path.join(cg, "memory.current")), _read(os.path.join(cg, "memory.max"))
    else:
        cg = _job_cgroup(paths, "memory", "memory.usage_in_bytes")
        cur = _read(os.path.join(cg, "memory.usage_in_bytes")) if cg else None
        lim = _read(os.path.join(cg, "memory.limit_in_bytes")) if cg else None
    gb = lambda v: (int(v) / 2**30) if v and v.isdigit() else None
    return gb(cur), gb(lim)


def node_mem_avail_gb() -> float | None:
    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return None


def sandboxes(url: str) -> tuple[int | None, int | None]:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/stats", timeout=5) as r:
            d = json.load(r)
        return d.get("inflight"), d.get("max_concurrent")
    except Exception:
        return None, None


def main() -> None:
    out, url = sys.argv[1], sys.argv[2]
    interval = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    paths = _cgroup_paths()
    fmt = lambda v: "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    if not os.path.exists(out):
        with open(out, "w") as f:
            f.write("ts,job_cpu_cores,job_cpu_limit_cores,job_mem_gb,job_mem_limit_gb,node_mem_avail_gb,load1,"
                    "ncpu_online,sandboxes,max_sandboxes\n")
    prev_t, prev_cpu = time.time(), cpu_usage_s(paths)
    while True:
        time.sleep(interval)
        try:
            t, c = time.time(), cpu_usage_s(paths)
            cores = (c - prev_cpu) / (t - prev_t) if (c is not None and prev_cpu is not None) else None
            prev_t, prev_cpu = t, c
            m, ml = mem(paths)
            sb, msb = sandboxes(url)
            row = [int(t), cores, cpu_limit_cores(paths), m, ml, node_mem_avail_gb(), os.getloadavg()[0],
                   os.cpu_count(), sb, msb]
            with open(out, "a") as f:
                f.write(",".join(fmt(v) for v in row) + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()
