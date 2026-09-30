#!/usr/bin/env python3
"""Submit a Miles-runtime job through slurm-compose (run locally, not on the cluster).

    python submit.py --cluster hel --partition interactive --nodes 1 --gpus 8 --hours 2 \
        --name bench-lora --owner stack [--pkg DIR]... [--env K=V]... -- bash '$SCOMPOSE_PKGS/miles_runtime/bench/run.sh' ...

The job runs COMMAND once per node (srun, 1 task per node) with this directory's
parent (`miles_runtime/`) exported as a package: `$SCOMPOSE_PKGS/miles_runtime/`.
Extra `--pkg` directories are exported next to it (`$SCOMPOSE_PKGS/<basename>/`).
Logs: <user_root>/miles/<owner>/joblogs/<jobid>-<project>-<name>.log (sbatch) and
<jobid>.<step>-<name>.log (srun). Needs ~/Desktop/code/cluster-tools (CLUSTER_TOOLS).
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME_DIR = HERE.parent
CLUSTER_TOOLS = Path(os.environ.get("CLUSTER_TOOLS", Path.home() / "Desktop/code/cluster-tools"))
_VENV_PY = CLUSTER_TOOLS / ".venv/bin/python"
if Path(sys.executable).resolve() != _VENV_PY.resolve() and _VENV_PY.exists():
    # cluster-tools' SSH plumbing (slurm-compose, paramiko, py>=3.11) lives in its venv
    os.execv(str(_VENV_PY), [str(_VENV_PY), __file__, *sys.argv[1:]])

CLUSTERS = {
    "hel": dict(hostname="nb-hel-cs-001-login-02.nvidia.com", account="nemotron_reason_math",
                user_root="/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/mbereket",
                sbatch="/cm/shared/apps/slurm/current/bin/sbatch", cpus=128, mem_gb=1500),
    "dfw": dict(hostname="cw-dfw-cs-001-login-02.nvidia.com", account="nemotron_reason_math",
                user_root="/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket",
                sbatch="/cm/shared/apps/slurm/current/bin/sbatch", cpus=128, mem_gb=1900),
    # interactive only (9 h), max 2 submitted jobs per user (pending count); $HOME over quota
    "aws-iad": dict(hostname="aws-iad-cs-002-login-02.nvidia.com", account="nemotron_reason_science",
                    user_root="/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket",
                    sbatch="/cm/shared/apps/slurm/current/bin/sbatch", cpus=192, mem_gb=1800),
}


def yaml_str(v: str) -> str:
    # slurm-compose re-parses values as YAML: quote so "false"/"8000"/"" survive as strings.
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cluster", required=True, choices=sorted(CLUSTERS))
    ap.add_argument("--partition", default="interactive")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=8, help="GPUs per node")
    ap.add_argument("--cpus", type=int, default=0, help="cpus per task (default: proportional to GPUs)")
    ap.add_argument("--mem", default="", help="e.g. 200G (default: proportional to GPUs)")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--owner", default="stack", help="miles/<owner>/ remote root (stack, path-a, path-b, ...)")
    ap.add_argument("--project", default="miles")
    ap.add_argument("--pkg", action="append", default=[], help="extra local dir exported to $SCOMPOSE_PKGS/<basename>")
    ap.add_argument("--env", action="append", default=[], help="K=V exported in the job")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd:
        ap.error("missing command")

    c = CLUSTERS[a.cluster]
    frac = a.gpus / 8 if a.gpus else 0.125
    cpus = a.cpus or max(8, int(c["cpus"] * frac) - (8 if a.gpus == 8 else 0))
    mem = a.mem or f"{max(64, int(c['mem_gb'] * frac))}G"
    owner_root = f"{c['user_root']}/miles/{a.owner}"
    hh = int(a.hours); mm = int(round((a.hours - hh) * 60))
    time = f"{hh:02d}:{mm:02d}:00"

    env = {
        "MILES_STACK_ROOT": f"{c['user_root']}/miles/stack",
        "MILES_OWNER_ROOT": owner_root,
        "CLUSTER_ALIAS": a.cluster,
        "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
    }
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v

    work = Path(tempfile.mkdtemp(prefix=f"miles-submit-{a.name}-"))
    sc_home = work / "sc-config"
    sc_home.mkdir()
    (sc_home / "hosts.toml").write_text(
        f'[{a.cluster}]\ntype = "ssh"\nhostname = "{c["hostname"]}"\nuser = "mbereket"\n'
        f'home_dir = "{owner_root}/.slurm-compose"\n\n[{a.cluster}.sbatch]\nbin = "{c["sbatch"]}"\n'
        f'account = "{c["account"]}"\ngpus_per_node = 8\n'
        f'partitions.gpu = {{partition = "{a.partition}", time = "{time}"}}\n'
        f'partitions.gpu_interactive = {{partition = "{a.partition}", time = "{time}"}}\n')
    pkgs = [RUNTIME_DIR] + [Path(p).resolve() for p in a.pkg]
    env_block = "\n".join(f"          {k}: {yaml_str(v)}" for k, v in env.items())
    command = " ".join(shlex.quote(x) if not x.startswith("$") and "$" not in x else x for x in cmd)
    gpu_line = f"    gpus_per_node: {a.gpus}\n" if a.gpus else ""
    step_gpu_line = f"        gpus_per_node: {a.gpus}\n" if a.gpus else ""
    yml = work / "job.yml"
    yml.write_text(
        "external_package_dirs:\n" + "".join(f"  - {p}\n" for p in pkgs) +
        f"jobs:\n  {a.name}:\n    time: {time}\n    nodes: {a.nodes}\n    ntasks_per_node: 1\n{gpu_line}"
        f"    cpus_per_task: {cpus}\n    mem: {mem}\n    requeue: false\n    steps:\n"
        f"      - step_type: srun\n        job_name: {a.name}\n        nodes: {a.nodes}\n        ntasks_per_node: 1\n"
        f"{step_gpu_line}        cpus_per_task: {cpus}\n        mem: {mem}\n        local_env:\n{env_block}\n"
        f"        command: {yaml_str(command)}\n")
    senv = dict(os.environ)
    senv.update(SCOMPOSE_CONFIG_HOME=str(sc_home), SCOMPOSE_PROJECT_NAME=a.project,
                SCOMPOSE_SBATCH_OUTPUT=f"{owner_root}/joblogs/%j-%x.log",
                SCOMPOSE_SRUN_OUTPUT=f"{owner_root}/joblogs/%j.%s-" + "${STEP_NAME}.log")
    sc = [str(CLUSTER_TOOLS / ".venv/bin/slurm-compose"), "-f", str(yml), "-H", a.cluster, "-p", a.partition,
          "-t", time, "-N", str(a.nodes), "--export-dir", str(work / "exports")]
    print(f"[{a.cluster}] {a.name}: {a.nodes}x{a.gpus} GPU, {cpus} cpu, {mem}, {time} on {a.partition}")
    print(f"  command: {command}")
    if a.dry_run:
        print(yml.read_text())
        return 0
    # the remote joblog dir must exist before the job starts
    sys.path.insert(0, str(CLUSTER_TOOLS))
    os.environ["SCOMPOSE_CONFIG_HOME"] = str(sc_home)
    import _common  # noqa: E402
    remote = _common.fs(a.cluster)
    for d in (f"{owner_root}/joblogs", f"{owner_root}/.slurm-compose/exports"):
        remote.makedirs(d, exist_ok=True)
    r = subprocess.run(sc, env=senv, capture_output=True, text=True)
    m = re.search(r"sbatch job (\d+) submitted", r.stdout + r.stderr)
    if not m:
        print((r.stdout + r.stderr)[-4000:], file=sys.stderr)
        return 1
    jid = m.group(1)
    print(f"  job {jid}  log: {owner_root}/joblogs/{jid}-{a.project}-{a.name}.<timestamp>.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
