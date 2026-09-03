#!/usr/bin/env python3
"""Materialize TMax-15k as a directory of Harbor tasks.

TMax (https://arxiv.org/abs/2606.23321) ships its RL data in two forms. The
Harbor hub dataset (``tmax/TMax-15K-Harbor``) carries a Dockerfile per task and
needs Docker to build images. The Hugging Face dataset used for the paper's own
training (``allenai/tmax-15k-open-instruct``) instead points every task at a
prebuilt public image on Docker Hub, which ``apptainer pull`` can fetch on a
Docker-less cluster. This script uses the second form and writes the task
directory layout the rest of this example consumes:

    <output>/manifest.json
    <output>/harbor/<task_id>/
        task.toml          [environment] docker_image, workdir; [agent]/[verifier] timeouts
        instruction.md     the task prompt
        tests/test.sh      the verifier (writes /logs/verifier/reward.txt)

Dependencies: huggingface_hub (download) and pyarrow (parquet); both are in the
Polar venv this example builds.

    python datasets/tmax15k.py --output <dir>              # all 14.6k tasks
    python datasets/tmax15k.py --output <dir> --limit 200  # first 200 (dataset order)

Selecting a training subset (``--n --seed``) happens later in prepare_tasks.py so
it works the same for any task directory, not just this one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HF_REPO = "allenai/tmax-15k-open-instruct"
PARQUET = "data/train-00000-of-00001.parquet"
TASK_DATA = "task-data.tar.gz"


def download(repo: str, cache_dir: str | None) -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    parquet = Path(hf_hub_download(repo, PARQUET, repo_type="dataset", cache_dir=cache_dir))
    tarball = Path(hf_hub_download(repo, TASK_DATA, repo_type="dataset", cache_dir=cache_dir))
    return parquet, tarball


def load_rows(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["messages", "env_config"])
    rows = []
    for record in table.to_pylist():
        env = record["env_config"] or {}
        messages = record["messages"] or []
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        rows.append({"task_id": env.get("task_id"), "image": env.get("image"), "prompt": user})
    return rows


def write_task(task_dir: Path, row: dict, staged: Path, opts: argparse.Namespace) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    instruction = (staged / "instruction.md").read_text(encoding="utf-8") if (staged / "instruction.md").is_file() else row["prompt"]
    (task_dir / "instruction.md").write_text(instruction.strip() + "\n", encoding="utf-8")
    tests_src = staged / "tests"
    if not (tests_src / "test.sh").is_file():
        raise SystemExit(f"{row['task_id']}: no tests/test.sh in task data")
    if (task_dir / "tests").exists():
        shutil.rmtree(task_dir / "tests")
    shutil.copytree(tests_src, task_dir / "tests")
    (task_dir / "tests" / "test.sh").chmod(0o755)
    # setup.sh is the recipe the prebuilt image was built from; kept for reference
    # only (the image already contains its effects), so it is not staged.
    if (staged / "setup.sh").is_file():
        shutil.copy2(staged / "setup.sh", task_dir / "image_setup.sh")
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.0"',
                "",
                "[task]",
                f'name = "tmax15k/{row["task_id"]}"',
                "",
                "[environment]",
                f'docker_image = "{row["image"]}"',
                f'workdir = "{opts.workdir}"',
                "",
                "[agent]",
                f"timeout_sec = {opts.agent_timeout}",
                "",
                "[verifier]",
                f"timeout_sec = {opts.verifier_timeout}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Task directory root to write (manifest.json + harbor/)")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--cache-dir", default=None, help="huggingface_hub cache dir (default: HF_HOME)")
    parser.add_argument("--parquet", default=None, help="Local copy of the dataset parquet (skips download)")
    parser.add_argument("--task-data", default=None, help="Local copy of task-data.tar.gz (skips download)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tasks in dataset order")
    parser.add_argument("--workdir", default="/app", help="Agent working directory inside the task container")
    parser.add_argument("--agent-timeout", type=int, default=1800, help="Agent wall-clock per attempt (s)")
    parser.add_argument("--verifier-timeout", type=int, default=600, help="test.sh budget (s)")
    parser.add_argument("--overwrite", action="store_true")
    opts = parser.parse_args()

    output = Path(opts.output).expanduser().resolve()
    harbor = output / "harbor"
    if output.exists() and any(output.iterdir()) and not opts.overwrite:
        raise SystemExit(f"{output} exists and is not empty (use --overwrite)")
    harbor.mkdir(parents=True, exist_ok=True)

    if opts.parquet and opts.task_data:
        parquet, tarball = Path(opts.parquet), Path(opts.task_data)
    else:
        print(f"downloading {opts.hf_repo}", file=sys.stderr)
        parquet, tarball = download(opts.hf_repo, opts.cache_dir)
    rows = [r for r in load_rows(parquet) if r["task_id"] and r["image"]]
    if opts.limit:
        rows = rows[: opts.limit]
    wanted = {r["task_id"] for r in rows}

    with tempfile.TemporaryDirectory(prefix="tmax15k-") as tmp:
        staging = Path(tmp)
        print(f"extracting task data for {len(wanted)} tasks", file=sys.stderr)
        with tarfile.open(tarball, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.split("/", 1)[0] in wanted]
            tar.extractall(staging, members=members, filter="data")
        manifest_tasks = []
        for i, row in enumerate(rows, 1):
            staged = staging / row["task_id"]
            if not staged.is_dir():
                print(f"skipping {row['task_id']}: not in task data", file=sys.stderr)
                continue
            task_dir = harbor / row["task_id"]
            write_task(task_dir, row, staged, opts)
            manifest_tasks.append(
                {
                    "directory": row["task_id"],
                    "source_id": row["task_id"],
                    "docker_image": row["image"],
                    "sha256": hashlib.sha256((task_dir / "instruction.md").read_bytes()).hexdigest()[:16],
                }
            )
            if i % 1000 == 0:
                print(f"  {i}/{len(rows)}", file=sys.stderr)

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "name": "tmax15k",
                "source": opts.hf_repo,
                "workdir": opts.workdir,
                "agent_timeout": opts.agent_timeout,
                "verifier_timeout": opts.verifier_timeout,
                "tasks": manifest_tasks,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {len(manifest_tasks)} tasks to {output}")


if __name__ == "__main__":
    main()
