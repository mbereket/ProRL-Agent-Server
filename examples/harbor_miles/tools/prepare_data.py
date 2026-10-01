"""Build a Miles prompt JSONL from a Harbor task directory.

    python prepare_data.py --tasks-dir <root>/harbor --out train.jsonl \
        [--ids-file ids.txt] [--agent opencode] \
        [--agent-import-path harbor_miles_agents.opencode_agents:BbhOpenCode] \
        [--opencode-config '{"compaction": {"auto": false}, "permission": {"task": "deny"}}'] \
        [--agent-env '{"OMP_NUM_THREADS": "4"}']

One row per task: {"prompt": [{"role": "user", "content": <instruction.md>}],
"metadata": {"instance_id": <task dir name>, "agent_name": ..., ...}}. The
prompt is nominal (the agent in the sandbox builds its own messages; the
session server records what the model actually saw); instance_id is what
selects the task on the Harbor agent servers (their HARBOR_TASKS_DIR must be
the same --tasks-dir).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ids-file", default="")
    p.add_argument("--agent", default="opencode")
    p.add_argument("--agent-import-path", default="")
    p.add_argument("--opencode-config", default="")
    p.add_argument("--agent-kwargs", default="", help="JSON merged into metadata.agent_kwargs")
    p.add_argument("--agent-env", default="", help="JSON {VAR: value}: extra env for the agent process (metadata.agent_env)")
    p.add_argument("--repeat", type=int, default=1, help="repeat the task list (small overfit sets)")
    a = p.parse_args()

    root = Path(a.tasks_dir)
    if a.ids_file:
        ids = [line.strip() for line in Path(a.ids_file).read_text().splitlines() if line.strip()]
    else:
        ids = sorted(d.name for d in root.iterdir() if (d / "task.toml").is_file())
    rows = []
    for tid in ids:
        task = root / tid
        if not (task / "task.toml").is_file():
            raise SystemExit(f"not a Harbor task dir: {task}")
        md = {"instance_id": tid, "agent_name": a.agent}
        if a.agent_import_path:
            md["agent_import_path"] = a.agent_import_path
        kwargs = json.loads(a.agent_kwargs) if a.agent_kwargs else {}
        if a.opencode_config:
            kwargs["opencode_config"] = json.loads(a.opencode_config)
        if kwargs:
            md["agent_kwargs"] = kwargs
        if a.agent_env:
            md["agent_env"] = {k: str(v) for k, v in json.loads(a.agent_env).items()}
        instruction = (task / "instruction.md").read_text()
        rows.append({"prompt": [{"role": "user", "content": instruction}], "metadata": md})
    with open(a.out, "w") as f:
        for _ in range(a.repeat):
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows) * a.repeat} rows ({len(rows)} tasks) -> {a.out}")


if __name__ == "__main__":
    main()
