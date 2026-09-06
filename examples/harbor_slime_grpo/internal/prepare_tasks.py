#!/usr/bin/env python3
"""Harbor task directory -> Slime prompt JSONL (or the list of images it needs).

    prepare_tasks.py --tasks-dir D --output-jsonl F [--image-dir SIFS] [--n N --seed S]
                     [--task-ids-file F] [--exclude-ids-file F] [--mount-root R]
    prepare_tasks.py --tasks-dir D --list-images [selection flags]   # "<docker_ref>\\t<sif>" per distinct image

Input is any directory of Harbor tasks: <root>/manifest.json + <root>/harbor/<task>/
or a plain directory whose subdirectories each hold a task.toml (what
`harbor datasets download --export` writes). Every task needs instruction.md,
task.toml with [environment] docker_image, and tests/test.sh (the verifier).
Optional: environment/files/setup.sh (staging run before the agent starts),
[environment] agent_path_prepend (put first on the agent's PATH after the
harness dirs), [agent]/[verifier] timeout_sec.

SIF names follow Harbor's singularity cache: "/" and ":" -> "_", ":latest" added
when the reference has no tag. With --image-dir the SIFs must already exist
(prepare_images.sh pulls them); missing ones are an error.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tomllib
from pathlib import Path

REQUIRED = ("task.toml", "instruction.md", "tests/test.sh")


def sif_name(docker_ref: str) -> str:
    if ":" not in docker_ref:
        docker_ref += ":latest"
    return docker_ref.replace("/", "_").replace(":", "_") + ".sif"


def discover(root: Path) -> list[tuple[Path, str]]:
    """(task_dir, source_id) pairs."""
    manifest = root / "manifest.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text()).get("tasks") or []
        return [(root / "harbor" / e["directory"], e.get("source_id", e["directory"])) for e in entries]
    return [(p.parent, p.parent.name) for p in sorted(root.rglob("task.toml"))]


def read_task(task_dir: Path) -> dict:
    for rel in REQUIRED:
        if not (task_dir / rel).is_file():
            sys.exit(f"ERROR: {task_dir}: missing {rel}")
    spec = tomllib.loads((task_dir / "task.toml").read_text())
    env = spec.get("environment", {})
    if not env.get("docker_image"):
        sys.exit(f"ERROR: {task_dir}: task.toml has no [environment] docker_image "
                 "(tasks that ship only a Dockerfile need their image built, pushed and named here)")
    return spec


def read_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    return {ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--list-images", action="store_true", help="print '<docker_ref>\\t<sif>' per distinct image and exit")
    p.add_argument("--output-jsonl")
    p.add_argument("--image-dir", help="SIF directory; every selected task's image must exist there")
    p.add_argument("--mount-root", help="host dir mounted as /harbor_data; task paths are emitted relative to it (default: --tasks-dir)")
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-ids-file")
    p.add_argument("--exclude-ids-file")
    p.add_argument("--default-agent-timeout", type=float, default=1800.0)
    p.add_argument("--default-verifier-timeout", type=float, default=600.0)
    a = p.parse_args()

    root = Path(a.tasks_dir).expanduser().resolve()
    tasks = discover(root)
    if not tasks:
        sys.exit(f"ERROR: no tasks under {root}")

    if not a.output_jsonl and not a.list_images:
        p.error("--output-jsonl is required (or --list-images)")

    keep, exclude = read_ids(a.task_ids_file), read_ids(a.exclude_ids_file)
    selected = [(d, s) for d, s in tasks
                if (not keep or s in keep or d.name in keep) and s not in exclude and d.name not in exclude]
    if keep:
        found = {s for _, s in selected} | {d.name for d, _ in selected}
        missing = sorted(k for k in keep if k not in found)
        if missing:
            sys.exit(f"ERROR: requested task ids not found: {missing}")
    if a.n is not None and a.n < len(selected):
        selected = sorted(random.Random(a.seed).sample(selected, a.n), key=lambda t: t[0].name)
    if not selected:
        sys.exit("ERROR: no tasks selected")
    if a.list_images:
        for ref in sorted({read_task(d)["environment"]["docker_image"] for d, _ in selected}):
            print(f"{ref}\t{sif_name(ref)}")
        return

    mount_root = Path(a.mount_root).expanduser().resolve() if a.mount_root else root
    lines, images = [], set()
    for task_dir, source_id in selected:
        spec = read_task(task_dir)
        env = spec["environment"]
        images.add(env["docker_image"])
        try:
            task_rel = task_dir.resolve().relative_to(mount_root).as_posix()
        except ValueError:
            sys.exit(f"ERROR: {task_dir} is not under --mount-root {mount_root}")
        lines.append(json.dumps({
            # Chat-formatted list: slime asserts list prompts when the model has an HF processor (Qwen3.5 is a VLM).
            "prompt": [{"role": "user", "content": (task_dir / "instruction.md").read_text().strip()}],
            "label": "",
            "metadata": {
                "instance_id": task_dir.name,
                "task_dir": task_dir.name,
                "task_rel": task_rel,
                "source_id": source_id,
                "image_sif": sif_name(env["docker_image"]),
                "workdir": env.get("workdir", "/app"),
                "path_prepend": (env["agent_path_prepend"].rstrip(":") + ":") if env.get("agent_path_prepend") else "",
                "has_setup": (task_dir / "environment" / "files" / "setup.sh").is_file(),
                "agent_timeout_sec": float(spec.get("agent", {}).get("timeout_sec", a.default_agent_timeout)),
                "verifier_timeout_sec": float(spec.get("verifier", {}).get("timeout_sec", a.default_verifier_timeout)),
            },
        }, ensure_ascii=False))

    if a.image_dir:
        missing = sorted(ref for ref in images if not (Path(a.image_dir) / sif_name(ref)).is_file())
        if missing:
            sys.exit(f"ERROR: {len(missing)} image(s) missing in {a.image_dir}; run prepare_images.sh {root}:\n  "
                     + "\n  ".join(missing))

    out = Path(a.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} tasks ({len(images)} distinct images) -> {out}")


if __name__ == "__main__":
    main()
