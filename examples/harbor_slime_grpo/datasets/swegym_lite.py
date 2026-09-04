#!/usr/bin/env python3
"""Materialize SWE-Gym (Lite by default) as a directory of Harbor tasks.

Follows the Harbor SWE-Gym adapter (harbor-framework/harbor, adapters/swegym,
Apache-2.0): same instruction (the GitHub issue text), same prebuilt images
(``xingyaoww/sweb.eval.x86_64.<owner>_s_<repo>-<issue>``), same verifier
(``swegym_test.sh.tmpl``: run FAIL_TO_PASS + PASS_TO_PASS with pytest inside
the image's ``testbed`` conda env, parse, write 0/1 to /logs/verifier/reward.txt).
The one difference is the output layout: the adapter emits a Dockerfile per
task, this writes the pullable image into ``task.toml`` so no image build is
needed (see prepare_tasks.py):

    <output>/manifest.json
    <output>/harbor/<instance_id>/
        task.toml            [environment] docker_image, workdir=/testbed, agent_path_prepend; timeouts
        instruction.md       problem statement
        tests/test.sh        verifier (hidden from the agent; uploaded after the run)
        tests/config.json    FAIL_TO_PASS / PASS_TO_PASS lists the parser reads

Splits: ``--dataset lite`` (SWE-Gym/SWE-Gym-Lite, 230 tasks; every image is
published) or ``--dataset full`` (SWE-Gym/SWE-Gym, 2438 tasks; 38 images are
missing upstream and fail at pull time). Dependency: ``datasets`` (in the venv).

    python datasets/swegym_lite.py --output <dir>              # all 230 lite tasks
    python datasets/swegym_lite.py --output <dir> --limit 10   # first 10 (dataset order)
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from textwrap import dedent

DATASETS = {"lite": "SWE-Gym/SWE-Gym-Lite", "full": "SWE-Gym/SWE-Gym"}
TEMPLATE = Path(__file__).resolve().with_name("swegym_test.sh.tmpl")
# Extensions swebench treats as non-test files when reading a test patch.
NON_TEST_EXTS = (".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml")


def as_list(v) -> list[str]:
    """FAIL_TO_PASS / PASS_TO_PASS arrive as a list, a JSON string, or a Python literal string."""
    if isinstance(v, list):
        return [str(x) for x in v]
    if not v:
        return []
    try:
        return [str(x) for x in json.loads(v)]
    except (json.JSONDecodeError, TypeError):
        return [str(x) for x in ast.literal_eval(v)]


def image_for(instance_id: str) -> str:
    return f"xingyaoww/sweb.eval.x86_64.{instance_id.replace('__', '_s_')}".lower()


def test_files_from_patch(test_patch: str) -> list[str]:
    files = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    return [f for f in files if not f.endswith(NON_TEST_EXTS)]


def problematic(test_name: str) -> bool:
    """Truncated parametrized ids or non-ASCII ids break pytest node-id selection."""
    if "[" in test_name and not test_name.endswith("]"):
        return True
    try:
        test_name.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def select_tests(fail_to_pass: list[str], pass_to_pass: list[str], test_patch: str) -> list[str]:
    """Node ids to run: FAIL_TO_PASS + PASS_TO_PASS, whole files where ids are unusable."""
    all_tests = fail_to_pass + pass_to_pass
    if not all_tests:
        return test_files_from_patch(test_patch)
    bad_files = {t.split("::")[0] for t in all_tests if problematic(t)}
    selected = sorted(bad_files)
    selected += [t for t in all_tests if not problematic(t) and t.split("::")[0] not in bad_files]
    return selected


EVAL_SCRIPT = dedent("""\
    set -o pipefail -x

    cd /testbed
    set +x
    source /opt/miniconda3/bin/activate
    conda activate testbed
    set -x
    set -u

    # Reset the files the test patch touches, then apply the test patch.
    @RESET_CMD@
    LOG_FILE=$(mktemp)
    export LOG_FILE
    exec 3>&1 4>&2
    exec > >(tee "$LOG_FILE") 2>&1
    echo @TEST_PATCH@ > /tmp/test_patch.diff
    git apply --check /tmp/test_patch.diff
    git apply /tmp/test_patch.diff

    # The agent edited the working tree in place; no submission diff to apply.
    echo ">>>>> Applied Patch (pred)"

    set +x
    # Batch to avoid OOM on large selections.
    BATCH_SIZE=40
    TESTS=(@TESTS@)
    if [ "${#TESTS[@]}" -gt "$BATCH_SIZE" ]; then
        TEST_OUTPUT_DIR=$(mktemp -d)
        BATCH_NUM=0
        for ((i=0; i<${#TESTS[@]}; i+=BATCH_SIZE)); do
            BATCH=("${TESTS[@]:i:BATCH_SIZE}")
            BATCH_NUM=$((BATCH_NUM + 1))
            pytest "${BATCH[@]}" > "$TEST_OUTPUT_DIR/batch_$BATCH_NUM.txt" 2>&1 || true
        done
        cat "$TEST_OUTPUT_DIR"/batch_*.txt
        rm -rf "$TEST_OUTPUT_DIR"
    else
        pytest "${TESTS[@]}" || true
    fi
    exec 1>&3 2>&4
    exec 3>&- 4>&-

    # Put the test files back.
    @RESET_CMD@
    """)


def eval_script(row: dict, fail_to_pass: list[str], pass_to_pass: list[str]) -> str:
    """The adapter's eval script. SWE-Gym repos have no upstream swebench specs, so
    the harness is the adapter's default: conda env ``testbed``, plain ``pytest``."""
    test_patch = row["test_patch"] or ""
    patched = re.findall(r"--- a/(.*)", test_patch)
    reset_cmd = f"git checkout {row['base_commit']} {' '.join(shlex.quote(f) for f in patched)}" if patched else ":"
    tests = select_tests(fail_to_pass, pass_to_pass, test_patch)
    return (EVAL_SCRIPT.replace("@RESET_CMD@", reset_cmd)
            .replace("@TEST_PATCH@", shlex.quote(test_patch))
            .replace("@TESTS@", " ".join(shlex.quote(t) for t in tests)))


def write_task(task_dir: Path, row: dict, template: str, opts: argparse.Namespace) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(exist_ok=True)
    instruction = dedent(row["problem_statement"]).strip() + "\n"
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    fail_to_pass, pass_to_pass = as_list(row["FAIL_TO_PASS"]), as_list(row["PASS_TO_PASS"])
    (task_dir / "tests" / "config.json").write_text(
        json.dumps({"instance_id": row["instance_id"], "repo": row["repo"], "version": row["version"],
                    "base_commit": row["base_commit"], "FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass},
                   indent=2))
    test_sh = template.replace("@EVAL_SCRIPT@", eval_script(row, fail_to_pass, pass_to_pass))
    (task_dir / "tests" / "test.sh").write_text(test_sh, encoding="utf-8")
    (task_dir / "tests" / "test.sh").chmod(0o755)
    (task_dir / "task.toml").write_text(
        "\n".join([
            'schema_version = "1.0"',
            "",
            "[task]",
            f'name = "swegym/{row["instance_id"]}"',
            "",
            "[environment]",
            f'docker_image = "{image_for(row["instance_id"])}"',
            'workdir = "/testbed"',
            # SWE-Gym images keep the repo's dependencies in this conda env; the agent's
            # python/pytest must come from it (prepended to PATH by the Polar template).
            'agent_path_prepend = "/opt/miniconda3/envs/testbed/bin"',
            "",
            "[agent]",
            f"timeout_sec = {opts.agent_timeout}",
            "",
            "[verifier]",
            f"timeout_sec = {opts.verifier_timeout}",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, help="Task directory root to write (manifest.json + harbor/)")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="lite")
    parser.add_argument("--cache-dir", default=None, help="datasets cache dir (default: HF_HOME)")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tasks in dataset order")
    parser.add_argument("--instance-id", action="append", default=None, help="Only these instance ids (repeatable)")
    parser.add_argument("--agent-timeout", type=int, default=1800, help="Agent wall-clock per attempt (s)")
    parser.add_argument("--verifier-timeout", type=int, default=600, help="test.sh budget (s)")
    parser.add_argument("--overwrite", action="store_true")
    opts = parser.parse_args()

    output = Path(opts.output).expanduser().resolve()
    harbor = output / "harbor"
    if output.exists() and any(output.iterdir()) and not opts.overwrite:
        raise SystemExit(f"{output} exists and is not empty (use --overwrite)")
    harbor.mkdir(parents=True, exist_ok=True)
    template = TEMPLATE.read_text(encoding="utf-8")

    from datasets import load_dataset

    name = DATASETS[opts.dataset]
    print(f"loading {name}", file=sys.stderr)
    rows = list(load_dataset(name, cache_dir=opts.cache_dir)["train"])
    if opts.instance_id:
        wanted = set(opts.instance_id)
        rows = [r for r in rows if r["instance_id"] in wanted]
        missing = wanted - {r["instance_id"] for r in rows}
        if missing:
            raise SystemExit(f"instance ids not in {name}: {sorted(missing)}")
    if opts.limit:
        rows = rows[: opts.limit]

    manifest_tasks = []
    for row in rows:
        task_dir = harbor / row["instance_id"]
        write_task(task_dir, row, template, opts)
        manifest_tasks.append({
            "directory": row["instance_id"],
            "source_id": row["instance_id"],
            "docker_image": image_for(row["instance_id"]),
            "sha256": hashlib.sha256((task_dir / "instruction.md").read_bytes()).hexdigest()[:16],
        })
    (output / "manifest.json").write_text(
        json.dumps({"dataset": name, "tasks": manifest_tasks}, indent=2), encoding="utf-8")
    print(f"{len(manifest_tasks)} tasks -> {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
