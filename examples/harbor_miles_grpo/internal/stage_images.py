#!/usr/bin/env python3
"""Stage the task SIFs of a task selection into this work root's image dir.

    stage_images.py --tasks-dir D --image-dir OWN [--shared-dir S]... [selection args of prepare_tasks.py]

For every selected task the expected SIF (prepare_tasks.sif_name of its docker_image) must end up in
OWN. Present -> nothing. Otherwise symlink a copy another work root already has, found under either
naming scheme in use on our clusters: the docker-ref name, or the task directory name
(<task_dir>.sif, older SWE-Gym staging). Prints the "<docker_ref>\\t<sif>" pairs still missing, which
the caller pulls, with the task dirs that use each image as a third column. Never writes outside OWN.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "harbor_slime_grpo" / "internal"))
from prepare_tasks import discover, read_ids, read_task, sif_name  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--shared-dir", action="append", default=[])
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-ids-file")
    p.add_argument("--exclude-ids-file")
    p.add_argument("--mount-root")  # accepted for argument compatibility with prepare_tasks.py
    a = p.parse_args()

    root = Path(a.tasks_dir).expanduser().resolve()
    keep, exclude = read_ids(a.task_ids_file), read_ids(a.exclude_ids_file)
    selected = [(d, s) for d, s in discover(root)
                if (not keep or s in keep or d.name in keep) and s not in exclude and d.name not in exclude]
    if a.n is not None and a.n < len(selected):
        selected = sorted(random.Random(a.seed).sample(selected, a.n), key=lambda t: t[0].name)

    own = Path(a.image_dir); own.mkdir(parents=True, exist_ok=True)
    shared = [Path(s) for s in a.shared_dir if s and Path(s).resolve() != own.resolve()]
    linked = present = 0
    missing: dict[str, str] = {}
    missing_tasks: dict[str, list[str]] = {}
    for task_dir, _ in selected:
        ref = read_task(task_dir)["environment"]["docker_image"]
        want = own / sif_name(ref)
        if want.exists() and want.stat().st_size > 0:
            present += 1
            continue
        candidates = [sdir / name for sdir in shared for name in (sif_name(ref), f"{task_dir.name}.sif")]
        hit = next((c for c in candidates if c.is_file() and c.stat().st_size > 0), None)
        if hit is not None:
            if want.is_symlink():
                want.unlink()
            os.symlink(hit, want)
            linked += 1
        else:
            missing[ref] = want.name
            missing_tasks.setdefault(ref, []).append(task_dir.name)
    print(f"images: {present} present, {linked} linked, {len(missing)} to pull", file=sys.stderr)
    for ref, sif in sorted(missing.items()):
        print(f"{ref}\t{sif}\t{','.join(missing_tasks[ref])}")


if __name__ == "__main__":
    main()
