#!/usr/bin/env python3
"""Shared helpers for the harbor_miles run-analysis tools: run-spec resolution, trial logs, job-log metrics.

Run spec (accepted by every tool):
  LOCAL_DIR             mirrored run dir (trials-*.jsonl, args-*.txt, gpu-*.csv, node-*.csv, chain.log; job logs in
                        LOCAL_DIR/joblogs/ or LOCAL_DIR/../joblogs/), e.g. made by fetch_run.sh
  CLUSTER:RUN_NAME      first existing of <user_root>/miles/runs/RUN (shared root), then the legacy
                        <user_root>/miles/{path-b,qwen27b,shared/hm}/runs/RUN
  CLUSTER:/abs/run/dir  any remote run dir
Remote specs are read over SFTP with cluster-tools (`_common.fs`, needs slurm_compose); if the current interpreter lacks
it, the tool re-executes itself once under the cluster-tools venv python ($CLUSTER_TOOLS/.venv/bin/python).

Job logs (<jobid>-hm-<name>[.<timestamp>].log and per-step <jobid>.<step>-<name>.log) are searched in the run's own
<HM_ROOT>/joblogs (= <run>/../../joblogs), then <user_root>/miles/joblogs and the legacy miles/{path-b,qwen27b,shared/hm}/
joblogs; only logs of the run's job ids are used (ids from trials-/args-/gpu-/agent_servers-/node-<jobid> file names and
`job=<id>` in chain.log / jobs.log). Local specs use LOCAL_DIR/joblogs (+ LOCAL_DIR/../joblogs), all *.log there if
none matches a job id.

Timestamps: job-log and nvidia-smi timestamps are node-local wall clock (measured 2026-10-01: US Pacific on dfw and hel,
consistent with the trial t_end epochs). They are parsed in $HM_LOG_TZ (default America/Los_Angeles; 'local' = this
machine's zone), so results do not depend on where the analysis runs.

CLI (used by fetch_run.sh):  hmruns.py resolve SPEC | jobids SPEC | joblogs SPEC [JOBID ...]
Provenance: consolidates the readers of DIAG overfit_steps.py, QWEN27B step_report.py / live_base_rates.py and PATH-B
profile_r2.py (2026-10-01).
"""
from __future__ import annotations

import ast
import collections
import datetime
import json
import os
import posixpath
import re
import sys

CLUSTER_TOOLS = os.path.expanduser(os.environ.get("CLUSTER_TOOLS", "~/Desktop/code/cluster-tools"))
CT_PYTHON = os.path.join(CLUSTER_TOOLS, ".venv", "bin", "python")

_MATH = "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/mbereket"
_SCIENCE = "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket"
USER_ROOTS = {"hel": _MATH, "dfw": _SCIENCE, "aws-iad": _SCIENCE, "ord": _SCIENCE, "draco": _SCIENCE}
# HM_ROOTs (each has runs/ and joblogs/) relative to the user root, searched in this order: shared root first, then legacy.
HM_ROOTS = ("miles", "miles/path-b", "miles/qwen27b", "miles/shared/hm")

OVERLONG = "SequenceLengthLimitExceeded"
METRIC_RE = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].* - (step|perf|rollout|eval) (\d+): (\{.*\})\s*$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
JOBLOG_RE = re.compile(r"(\d+)[.-].*\.log$")
NAN = float("nan")


# ----------------------------------------------------------------------------------------------------------- time
def _log_tz():
    name = os.environ.get("HM_LOG_TZ", "America/Los_Angeles")
    if name == "local":
        return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 - missing tz database: fall back to local time
        print(f"warning: time zone {name!r} unavailable; parsing log timestamps in local time", file=sys.stderr)
        return None


LOG_TZ = _log_tz()


def parse_ts(s: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> float:
    """Node-local log timestamp -> epoch seconds."""
    return datetime.datetime.strptime(s, fmt).replace(tzinfo=LOG_TZ).timestamp()


def fmt_ts(t: float, fmt: str = "%H:%M") -> str:
    """Epoch seconds -> node-local wall clock string."""
    return datetime.datetime.fromtimestamp(t, LOG_TZ).strftime(fmt)


# ----------------------------------------------------------------------------------------------------------- remote fs
_FS: dict = {}


def remote_fs(cluster: str):
    """fsspec-like SFTP filesystem for CLUSTER (cluster-tools). Re-executes under the cluster-tools venv if needed."""
    if cluster not in _FS:
        try:
            if CLUSTER_TOOLS not in sys.path:
                sys.path.insert(0, CLUSTER_TOOLS)
            from _common import fs
            _FS[cluster] = fs(cluster)
        except ImportError as e:
            if os.path.exists(CT_PYTHON) and not os.environ.get("HM_REEXEC") and os.path.isfile(sys.argv[0]):
                print(f"(remote run spec: re-running under {CT_PYTHON})", file=sys.stderr)
                sys.stdout.flush()
                sys.stderr.flush()
                os.environ["HM_REEXEC"] = "1"
                os.execv(CT_PYTHON, [CT_PYTHON, os.path.abspath(sys.argv[0]), *sys.argv[1:]])
            sys.exit(f"remote run specs need cluster-tools with slurm_compose ({e}); run with {CT_PYTHON} or set CLUSTER_TOOLS")
    return _FS[cluster]


# ----------------------------------------------------------------------------------------------------------- runs
class Run:
    """A run dir, local or remote. Paths returned by its methods are full paths in the run's filesystem."""

    def __init__(self, spec: str, cluster: str | None, path: str):
        self.spec, self.cluster = spec, cluster
        self.path = path.rstrip("/") or "/"
        self.name = posixpath.basename(self.path) if cluster else os.path.basename(self.path)

    def __repr__(self):
        return f"Run({self.key})"

    @property
    def key(self) -> str:
        return f"{self.cluster}:{self.path}" if self.cluster else self.path

    def join(self, *parts: str) -> str:
        return (posixpath if self.cluster else os.path).join(self.path, *parts)

    def exists(self, path: str) -> bool:
        return remote_fs(self.cluster).exists(path) if self.cluster else os.path.exists(path)

    def listdir(self, path: str | None = None) -> list[str]:
        """Sorted full paths of the entries of `path` (default: the run dir); [] if it does not exist."""
        path = path or self.path
        try:
            if self.cluster:
                return sorted(posixpath.join(path, posixpath.basename(p.rstrip("/")))
                              for p in remote_fs(self.cluster).ls(path, detail=False))
            return sorted(os.path.join(path, n) for n in os.listdir(path))
        except (FileNotFoundError, NotADirectoryError, OSError):
            return []

    def files(self, pattern: str, path: str | None = None) -> list[str]:
        """Entries of `path` (default: run dir) whose basename fully matches the regex `pattern`."""
        rx = re.compile(pattern)
        return [p for p in self.listdir(path) if rx.fullmatch(os.path.basename(p))]

    def read_bytes(self, path: str) -> bytes:
        if self.cluster:
            with remote_fs(self.cluster).open(path, "rb") as f:
                try:
                    f.prefetch()  # paramiko: pipelined reads (sequential 32 KB round trips otherwise)
                except Exception:  # noqa: BLE001
                    pass
                return f.read()
        with open(path, "rb") as f:
            return f.read()

    def read_text(self, path: str) -> str:
        return self.read_bytes(path).decode(errors="replace")

    def jobids(self) -> list[str]:
        """Slurm job ids that wrote into this run dir, ascending."""
        ids = set()
        for p in self.listdir():
            b = os.path.basename(p)
            m = re.fullmatch(r"(?:trials|args|gpu|agent_servers)-(\d+)\.\w+", b) or re.match(r"node-(\d+)-", b)
            if m:
                ids.add(m.group(1))
            elif b in ("chain.log", "jobs.log"):
                for line in self.read_text(p).splitlines():
                    m = re.search(r"\bjob(?:id)?[=: ]+(\d+)", line)
                    if m:
                        ids.add(m.group(1))
                    elif b == "jobs.log":  # unknown format: any 5-9 digit token (not epochs, not dates)
                        ids.update(re.findall(r"\b(\d{5,9})\b", line))
        return sorted(ids, key=int)

    def joblog_dirs(self) -> list[str]:
        if self.cluster:
            cands = [posixpath.join(posixpath.dirname(posixpath.dirname(self.path)), "joblogs")]
            if self.cluster in USER_ROOTS:
                cands += [f"{USER_ROOTS[self.cluster]}/{hm}/joblogs" for hm in HM_ROOTS]
        else:
            cands = [os.path.join(self.path, "joblogs"), os.path.join(os.path.dirname(self.path), "joblogs")]
        out = []
        for c in cands:
            if c not in out and self.exists(c):
                out.append(c)
        return out


def resolve_run(spec: str, must_exist: bool = True) -> Run | None:
    """LOCAL_DIR | CLUSTER:RUN_NAME | CLUSTER:/abs/dir -> Run (None if not found and must_exist=False)."""
    if os.path.isdir(spec):
        return Run(spec, None, os.path.abspath(spec))
    m = re.fullmatch(r"([A-Za-z0-9_.-]+):(.+)", spec)
    if not m:
        if must_exist:
            sys.exit(f"run spec {spec!r}: not a local dir and not CLUSTER:RUN_NAME / CLUSTER:/abs/dir")
        return None
    cluster, rest = m.groups()
    if rest.startswith("/"):
        cands = [rest]
    elif cluster in USER_ROOTS:
        cands = [f"{USER_ROOTS[cluster]}/{hm}/runs/{rest.strip('/')}" for hm in HM_ROOTS]
    else:
        sys.exit(f"run spec {spec!r}: unknown cluster {cluster!r} for a run name (known: {', '.join(USER_ROOTS)}); "
                 f"use CLUSTER:/abs/run/dir")
    fs = remote_fs(cluster)
    for c in cands:
        if fs.exists(c):
            return Run(spec, cluster, c)
    if must_exist:
        sys.exit(f"run spec {spec!r}: not found on {cluster} (tried {', '.join(cands)})")
    return None


# ----------------------------------------------------------------------------------------------------------- trials
def row_split(r: dict) -> str:
    return r.get("split") or "train"  # rows written before split tagging are train


def read_trials(run: Run, jobids=None, split: str | None = None, include_infra: bool = True) -> list[dict]:
    """Rows of trials-*.jsonl (or trials-<jobid>.jsonl for the given job ids), sorted by t_end. Non-JSON lines are
    skipped (a partially written last line on a live run). split: 'train' / 'eval' / None or 'all' = all."""
    paths = run.files(r"trials-.*\.jsonl")
    if jobids:
        want = {f"trials-{j}.jsonl" for j in jobids}
        paths = [p for p in paths if os.path.basename(p) in want]
    rows = []
    for p in paths:
        for line in run.read_text(p).splitlines():
            if not line.lstrip().startswith("{"):
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                rows.append(r)
    if split and split != "all":
        rows = [r for r in rows if row_split(r) == split]
    if not include_infra:
        rows = [r for r in rows if not r.get("infra_failure")]
    rows.sort(key=lambda r: r.get("t_end") or 0)
    return rows


def spec_success(r: dict) -> float:
    """Success with truncation counted as failure: 0 for SequenceLengthLimitExceeded, else the reward."""
    return 0.0 if r.get("exit_status") == OVERLONG else float(r.get("reward") or 0.0)


def is_overlong(r: dict) -> bool:
    return r.get("exit_status") == OVERLONG


def by_task(rows) -> dict[str, list[dict]]:
    out = collections.defaultdict(list)
    for r in rows:
        out[r.get("instance_id")].append(r)
    return out


# ----------------------------------------------------------------------------------------------------------- job logs
def find_joblogs(run: Run, jobids=None) -> list[str]:
    """Job-log paths of the run's job ids (or the given ones), ordered by (job id, name)."""
    ids = {str(j) for j in (jobids or run.jobids())}
    out, seen = [], set()
    dirs = run.joblog_dirs()
    for d in dirs:
        for p in run.listdir(d):
            b = os.path.basename(p)
            m = JOBLOG_RE.fullmatch(b)
            if m and m.group(1) in ids and b not in seen:
                seen.add(b)
                out.append(p)
    if not out and not run.cluster and not jobids:  # local mirror with foreign naming: everything in <run>/joblogs
        out = [p for p in run.listdir(run.join("joblogs")) if p.endswith(".log")]

    def key(p):
        m = JOBLOG_RE.fullmatch(os.path.basename(p))
        return (int(m.group(1)) if m else 0, os.path.basename(p))
    return sorted(out, key=key)


def read_joblogs(run: Run, paths=None, jobids=None, local_paths=None, quiet: bool = False) -> str:
    """Concatenated, ANSI-stripped text of the run's job logs (`local_paths` = explicit local files override)."""
    if local_paths:
        texts = [open(p, "rb").read().decode(errors="replace") for p in local_paths]
        names = list(local_paths)
    else:
        paths = paths if paths is not None else find_joblogs(run, jobids)
        texts = [run.read_text(p) for p in paths]
        names = paths
    if not quiet:
        print(f"[joblogs: {', '.join(os.path.basename(n) for n in names) or 'NONE FOUND'}]", file=sys.stderr)
    return ANSI_RE.sub("", "\n".join(texts))


def literal_dict(s: str):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    # metric dicts with nan/inf values (repr `nan`, `inf`): inf -> 1e999 literal, nan -> sentinel string -> float nan
    s2 = re.sub(r"(?<=[:,\[] )(-?)(nan|inf)(?=\s*[,}\]])", lambda m: m.group(1) + "1e999" if m.group(2) == "inf" else "'__nan__'", s)
    d = ast.literal_eval(s2)
    return {k: (NAN if v == "__nan__" else v) for k, v in d.items()} if isinstance(d, dict) else d


Joblog = collections.namedtuple("Joblog", "metrics step_t evals")


def iter_metric_lines(text: str, kinds=("step", "perf", "rollout", "eval")):
    """(timestamp str, kind, n, dict, line) for every `[ts ...] ... - KIND N: {...}` line."""
    for line in text.splitlines():
        if " - " not in line or ": {" not in line:
            continue
        m = METRIC_RE.search(line)
        if not m or m.group(2) not in kinds:
            continue
        try:
            d = literal_dict(m.group(4))
        except (ValueError, SyntaxError):
            continue
        if isinstance(d, dict):
            yield m.group(1), m.group(2), int(m.group(3)), d, line


def parse_joblog(text: str) -> Joblog:
    """One pass: metrics {n: merged step/perf/rollout dict}, step_t {n: epoch of the trainer's log_utils `step n` line},
    evals {n: (timestamp str, dict)}. Later lines win (resumed chunks re-logging a step override the earlier copy)."""
    met, ts, ev = collections.defaultdict(dict), {}, {}
    for t, kind, n, d, line in iter_metric_lines(text):
        if kind == "eval":
            ev[n] = (t, d)
            continue
        met[n].update(d)
        if kind == "step" and "log_utils" in line:
            ts[n] = parse_ts(t)
    return Joblog(dict(met), ts, ev)


def parse_metrics(text: str, kinds=("step", "perf", "rollout", "eval")) -> dict[int, dict]:
    met = collections.defaultdict(dict)
    for _t, _k, n, d, _l in iter_metric_lines(text, kinds):
        met[n].update(d)
    return dict(met)


def step_times(text: str) -> dict[int, float]:
    return parse_joblog(text).step_t


def assign_steps(trials: list[dict], step_t: dict[int, float]) -> list[dict]:
    """Set r['step'] = first step n (ascending) with t_end <= t_step[n], i.e. t_end in (t_step[n-1], t_step[n]];
    trials after the last logged step get last + 1 (in progress; 0 if no step is logged)."""
    order = sorted(step_t)
    tail = (order[-1] + 1) if order else 0
    for r in trials:
        te = r.get("t_end") or 0
        r["step"] = next((n for n in order if te <= step_t[n]), tail)
    return trials


# ----------------------------------------------------------------------------------------------------------- misc
def read_args(run: Run, jobid=None) -> list[str]:
    """Tokens of args-<jobid>.txt (default: the newest job's), [] if absent."""
    paths = run.files(r"args-.*\.txt")
    if jobid is not None:
        paths = [p for p in paths if os.path.basename(p) == f"args-{jobid}.txt"]
    paths.sort(key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    return run.read_text(paths[-1]).split() if paths else []


def arg_value(tokens: list[str], flag: str, default=None):
    return tokens[tokens.index(flag) + 1] if flag in tokens and tokens.index(flag) + 1 < len(tokens) else default


def parse_gpu_csv(text: str) -> list[tuple[float, int, int]]:
    """nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits
    -> [(epoch, gpu index, memory used MiB)]."""
    out = []
    for x in text.splitlines():
        q = [y.strip() for y in x.split(",")]
        if len(q) >= 3 and q[0][:4].isdigit():
            try:
                out.append((parse_ts(q[0][:19], "%Y/%m/%d %H:%M:%S"), int(q[1]), int(q[2])))
            except ValueError:
                pass
    return out


def num(d: dict, k: str, default: float = NAN) -> float:
    v = d.get(k)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 2 or a[0] not in ("resolve", "jobids", "joblogs"):
        print(__doc__, file=sys.stderr)
        return 2
    run = resolve_run(a[1])
    if a[0] == "resolve":
        print(run.path)
    elif a[0] == "jobids":
        print("\n".join(run.jobids()))
    else:
        print("\n".join(find_joblogs(run, a[2:] or None)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
