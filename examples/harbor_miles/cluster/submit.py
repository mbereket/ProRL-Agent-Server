#!/usr/bin/env python3
"""Submit a harbor_miles job through slurm-compose (run locally, cluster-tools venv).

    ~/Desktop/code/cluster-tools/.venv/bin/python cluster/submit.py \
        --cluster hel --partition interactive --nodes 1 --gpus 1 --cpus 48 --mem 400G \
        --hours 4 --name hv1 -- tools/hv1_job.sh [args...]

The example directory is the slurm-compose package (uploaded with every
submission, so no git push is needed to iterate); the command runs once per
node (ntasks_per_node=1) as ``bash $SCOMPOSE_PKGS/harbor_miles/<script> args``
with CLUSTER, JOB_NAME and any ``--env K=V`` set. Job logs land in
``<user root>/miles/joblogs/%j-<project>-<name>.log`` (+ per-step ``%j.%s-<name>.log``); run dirs in
``<user root>/miles/runs/<RUN_NAME>/`` (cluster/clusters.sh HM_RUNS_ROOT). HM_ROOT (setup: Harbor venv, toolchains, caches)
stays per cluster (ROOTS below).

For ``launch/node_entry.sh CONFIG`` submissions the config is dry-rendered first (tools/dry_render.sh with this cluster and
--nodes): a config that fails its layout checks (node count, cap vs trainer fit, single-node cluster) is never queued.
``--no-render`` skips that.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
CLUSTER_TOOLS = Path(os.environ.get("CLUSTER_TOOLS", Path.home() / "Desktop/code/cluster-tools"))
PROJECT = "hm"
# Local staging for slurm-compose exports: OUTSIDE the package (the package is uploaded).
EXPORT_ROOT = Path(os.environ.get("HM_EXPORT_ROOT", Path.home() / ".cache/harbor_miles/slurm-compose-exports"))

ROOTS = {
    "hel": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/mbereket/miles/path-b",
    "dfw": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/path-b",
    # FLEET-staged clusters: shared writable root next to miles/shared. aws-iad: interactive only (2 submitted jobs/user,
    # 9 h), --mem <= 1700G. OCI clusters (ord, draco; A100): <= 30 cpus/GPU and < node_total/8 mem per GPU -> --cpus 240
    # --mem 1680G for 8 GPUs; ord interactive_singlenode / draco interactive allow ONE running job per user.
    "aws-iad": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/shared/hm",
    "ord": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/shared/hm",
    "draco": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket/miles/shared/hm",
}

# <user root> per cluster: run dirs (<user root>/miles/runs) and job logs (<user root>/miles/joblogs) for every run.
USER_ROOTS = {
    "hel": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_math/users/mbereket",
    "dfw": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket",
    "aws-iad": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket",
    "ord": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket",
    "draco": "/lustre/fsw/portfolios/nemotron/projects/nemotron_reason_science/users/mbereket",
}

TEMPLATE = """\
external_package_dirs:
  - ..
{extra_pkgs}jobs:
  {name}:
    time: "{time}"
    nodes: {nodes}
    ntasks_per_node: 1
    gpus_per_node: {gpus}
    cpus_per_task: {cpus}
    mem: {mem}
    requeue: false
    steps:
      - step_type: srun
        job_name: {name}
        nodes: {nodes}
        ntasks_per_node: 1
        gpus_per_node: {gpus}
        cpus_per_task: {cpus}
        mem: {mem}
        local_env:
{env_block}
        command: bash "${{SCOMPOSE_PKGS}}/{pkg}/{script}" {args}
"""


def _q(v: str) -> str:
    # Values are re-parsed as YAML by slurm-compose: quote everything.
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def ensure_remote_dirs(alias: str, *dirs: str) -> None:
    os.environ.setdefault("SCOMPOSE_CONFIG_HOME", str(HERE / "sc-config"))
    sys.path.insert(0, str(CLUSTER_TOOLS))
    import _common  # noqa: E402

    remote = _common.fs(alias)
    for d in dirs:
        remote.makedirs(d, exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", required=True, choices=sorted(ROOTS))
    p.add_argument("--partition", default="interactive")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--cpus", type=int, default=96)
    p.add_argument("--mem", default="1200G")
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--name", required=True)
    p.add_argument("--env", action="append", default=[], help="K=V exported in the job")
    p.add_argument("--extra-pkg", action="append", default=[],
                   help="extra local dir uploaded as a package ($SCOMPOSE_PKGS/<basename>), e.g. miles_runtime")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-render", action="store_true", help="skip the local dry render of a node_entry.sh config")
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = [x for x in a.args if x != "--"]

    root = os.environ.get("HM_SUBMIT_ROOT") or ROOTS[a.cluster]   # HM_ROOT: setup root (Harbor venv, toolchains, caches)
    joblogs = f"{USER_ROOTS[a.cluster]}/miles/joblogs"
    env = {"CLUSTER": a.cluster, "JOB_NAME": a.name, "HM_ROOT": root}
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    h = int(a.hours)
    m = int(round((a.hours - h) * 60))
    yml = TEMPLATE.format(
        name=a.name, time=f"{h:02d}:{m:02d}:00", nodes=a.nodes, gpus=a.gpus, cpus=a.cpus, mem=a.mem,
        env_block="\n".join(f"          {k}: {_q(v)}" for k, v in env.items()),
        pkg=EXAMPLE.name, script=a.script, args=" ".join(shlex.quote(x) for x in args),
        extra_pkgs="".join(f"  - {Path(d).resolve()}\n" for d in a.extra_pkg),
    )
    if a.script.endswith("node_entry.sh") and args and not a.no_render:
        extra = [kv for kv in a.env if not kv.startswith("HM_ROOT=")]
        hmr = [kv.split("=", 1)[1] for kv in a.env if kv.startswith("HM_ROOT=")]
        cmd = ["bash", str(EXAMPLE / "tools/dry_render.sh"), "--cluster", a.cluster, "--nodes", str(a.nodes),
               *(["--hm-root", hmr[-1]] if hmr else []), args[0], *extra]
        r = subprocess.run(cmd, cwd=EXAMPLE, capture_output=True, text=True)
        print(r.stdout.strip())
        if r.returncode != 0:
            print("\n".join(l for l in r.stderr.splitlines() if "hm_derive_config" in l or "FATAL" in l or "hm_load_config" in l)
                  or r.stderr.strip()[-2000:])
            print("NOT submitted: the config does not render for this cluster / node count (see above; --no-render skips)")
            return 1
    rendered = HERE / "rendered"
    rendered.mkdir(exist_ok=True)
    yml_path = rendered / f"{a.name}.yml"
    yml_path.write_text(yml)
    cmd = [str(CLUSTER_TOOLS / ".venv/bin/slurm-compose"), "-f", str(yml_path.relative_to(HERE)), "-H", a.cluster,
           "-p", a.partition, "-t", f"{h:02d}:{m:02d}:00", "-N", str(a.nodes),
           "--export-dir", str(EXPORT_ROOT)]
    senv = dict(os.environ)
    senv["SCOMPOSE_CONFIG_HOME"] = str(HERE / "sc-config")
    senv["SCOMPOSE_SBATCH_OUTPUT"] = f"{joblogs}/%j-%x.log"
    senv["SCOMPOSE_SRUN_OUTPUT"] = f"{joblogs}/%j.%s-${{STEP_NAME}}.log"
    senv["SCOMPOSE_PROJECT_NAME"] = PROJECT
    print("$", " ".join(shlex.quote(c) for c in cmd))
    if a.dry_run:
        print(yml)
        return 0
    ensure_remote_dirs(a.cluster, joblogs, f"{USER_ROOTS[a.cluster]}/miles/runs", f"{root}/.slurm-compose/exports")
    r = subprocess.run(cmd, cwd=HERE, env=senv, capture_output=True, text=True)
    text = r.stdout + r.stderr
    mm = re.search(r"sbatch job (\d+) submitted", text)
    if not mm:
        print(text[-4000:])
        return 1
    print(f"submitted {a.cluster} job {mm.group(1)} ({a.name}); logs: {joblogs}/{mm.group(1)}-{PROJECT}-{a.name}.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
