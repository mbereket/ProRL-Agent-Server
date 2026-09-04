#!/usr/bin/env python3
"""Harbor task directory -> Slime prompt JSONL + image list.

Input is any directory of Harbor tasks: either ``<root>/manifest.json`` +
``<root>/harbor/<task>/`` (what datasets/*.py and eval pipelines write) or a
plain directory whose subdirectories each hold a ``task.toml``. Every task must
have ``instruction.md``, ``task.toml`` with ``[environment] docker_image``, and
``tests/test.sh`` (the verifier). ``environment/files/`` and its ``setup.sh``
are optional staging the runtime applies before the agent starts;
``[environment] agent_path_prepend`` (optional) is put first on the agent's PATH
after the harness dirs (SWE-Gym images: the repo's conda env).

Output:
  --output-jsonl   one line per task for the Polar/Slime bridge (--input-key prompt,
                   --metadata-key metadata); metadata carries everything the Polar
                   task template needs (image SIF, task path, workdir, timeouts).
  --output-images  "<docker_ref>\\t<sif_name>" per unique image, for prepare_images.sh.

Subset selection: --n/--seed samples tasks at random; --task-ids/--task-ids-file
and --exclude-ids-file pick or drop tasks by directory name or manifest source_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from pathlib import Path


def sif_name_for(docker_ref: str) -> str:
    if docker_ref.endswith(".sif"):
        return Path(docker_ref).name
    if "@sha256:" in docker_ref:
        return f"sha256-{docker_ref.rsplit('@sha256:', 1)[1][:32]}.sif"
    return f"ref-{hashlib.sha256(docker_ref.encode()).hexdigest()[:32]}.sif"


def read_ids(path: str | None) -> set[str]:
    if not path:
        return set()
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")}


def discover(root: Path) -> list[tuple[Path, str]]:
    """(task_dir, source_id) pairs."""
    manifest = root / "manifest.json"
    if manifest.is_file():
        entries = json.loads(manifest.read_text()).get("tasks") or []
        return [(root / "harbor" / e["directory"], e.get("source_id", e["directory"])) for e in entries]
    return [(p.parent, p.parent.name) for p in sorted(root.rglob("task.toml"))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", required=True, help="manifest.json + harbor/, or a directory of task dirs")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-images", required=True)
    parser.add_argument("--mount-root", default=None,
                        help="Host directory mounted into runtimes as /harbor_data; task paths are emitted "
                             "relative to it (default: --tasks-dir)")
    parser.add_argument("--n", type=int, default=None, help="Sample this many tasks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task-ids", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--exclude-ids-file", default=None)
    parser.add_argument("--default-agent-timeout", type=float, default=1800.0)
    parser.add_argument("--default-verifier-timeout", type=float, default=600.0)
    args = parser.parse_args()

    root = Path(args.tasks_dir).expanduser().resolve()
    mount_root = Path(args.mount_root).expanduser().resolve() if args.mount_root else root
    tasks = discover(root)
    if not tasks:
        raise SystemExit(f"No tasks under {root}")

    keep = set(args.task_ids or []) | read_ids(args.task_ids_file)
    exclude = read_ids(args.exclude_ids_file)
    selected = [(d, s) for d, s in tasks
                if (not keep or s in keep or d.name in keep) and s not in exclude and d.name not in exclude]
    if keep:
        found = {s for _, s in selected} | {d.name for d, _ in selected}
        missing = sorted(k for k in keep if k not in found)
        if missing:
            raise SystemExit(f"Requested task ids not found: {missing}")
    if args.n is not None and args.n < len(selected):
        selected = random.Random(args.seed).sample(selected, args.n)
        selected.sort(key=lambda t: t[0].name)
    if not selected:
        raise SystemExit("No tasks selected")

    images: dict[str, str] = {}
    lines: list[str] = []
    for task_dir, source_id in selected:
        toml_path = task_dir / "task.toml"
        instruction_path = task_dir / "instruction.md"
        test_sh = task_dir / "tests" / "test.sh"
        for required in (toml_path, instruction_path, test_sh):
            if not required.is_file():
                raise SystemExit(f"{task_dir.name}: missing {required.relative_to(task_dir)}")
        spec = tomllib.loads(toml_path.read_text())
        env = spec.get("environment", {})
        docker_ref = env.get("docker_image")
        if not docker_ref:
            hint = " (has environment/Dockerfile: build and push it, then set docker_image)" \
                if (task_dir / "environment" / "Dockerfile").is_file() else ""
            raise SystemExit(f"{task_dir.name}: task.toml has no [environment] docker_image{hint}")
        sif = sif_name_for(docker_ref)
        images[docker_ref] = sif
        try:
            task_rel = task_dir.resolve().relative_to(mount_root).as_posix()
        except ValueError:
            raise SystemExit(f"{task_dir} is not under --mount-root {mount_root}") from None
        lines.append(json.dumps({
            # Chat-formatted list: slime asserts list prompts when the model has an
            # HF processor (Qwen3.5 checkpoints are VLMs).
            "prompt": [{"role": "user", "content": instruction_path.read_text().strip()}],
            "label": "",
            "metadata": {
                "instance_id": task_dir.name,
                "task_dir": task_dir.name,
                "task_rel": task_rel,
                "source_id": source_id,
                "image_sif": sif,
                "workdir": env.get("workdir", "/app"),
                # Prepended to the agent's PATH (e.g. the image's conda env); "" when unset.
                "path_prepend": (env["agent_path_prepend"].rstrip(":") + ":") if env.get("agent_path_prepend") else "",
                "has_setup": (task_dir / "environment" / "files" / "setup.sh").is_file(),
                "agent_timeout_sec": float(spec.get("agent", {}).get("timeout_sec", args.default_agent_timeout)),
                "verifier_timeout_sec": float(spec.get("verifier", {}).get("timeout_sec", args.default_verifier_timeout)),
            },
        }, ensure_ascii=False))

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    Path(args.output_images).write_text("".join(f"{ref}\t{sif}\n" for ref, sif in sorted(images.items())))
    print(f"{len(lines)} tasks -> {out}")
    print(f"{len(images)} unique image(s) -> {args.output_images}")


if __name__ == "__main__":
    main()
