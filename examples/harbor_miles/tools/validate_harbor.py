"""Drive Harbor agent servers directly (no Miles) to validate the sandbox layer.

    python validate_harbor.py trials --tasks-dir D --task-ids a,b,c --attempts 1 \
        --base-url http://host:30600/v1 --model openai/qwen35-9b --out results.jsonl
    python validate_harbor.py nop --tasks-dir D --task-ids a --n 32 --out nop.jsonl
    python validate_harbor.py flush --tasks-dir D --task-ids a --n 2 --after 120 ...

Servers come from HARBOR_AGENT_SERVERS / HARBOR_AGENT_SERVERS_FILE (see
miles_side/hm_dispatch.py), exactly as in training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from miles_side.hm_dispatch import POOL  # noqa: E402


def _task_ids(a: argparse.Namespace) -> list[str]:
    if a.task_ids:
        return [t for t in a.task_ids.split(",") if t]
    return sorted(p.name for p in Path(a.tasks_dir).iterdir() if (p / "task.toml").exists())


def _request(a: argparse.Namespace, task_id: str, agent: str) -> dict:
    req = {
        "base_url": a.base_url,
        "model": a.model,
        "sampling_params": {"temperature": a.temperature, "top_p": 1.0, "max_tokens": a.max_tokens},
        "instance_id": task_id,
        "agent_name": agent,
        "max_seq_len": a.max_seq_len,
    }
    if agent == "sleep":
        req["agent_name"] = "nop"
        req["agent_import_path"] = "harbor_miles_agents.test_agents:SleepAgent"
        req["agent_kwargs"] = {"sleep_sec": 900}
    elif agent == "opencode" and a.agent_import_path and "bbh" in a.agent_import_path.lower():
        req["agent_import_path"] = a.agent_import_path
        req["agent_kwargs"] = {
            "opencode_config": {"compaction": {"auto": False}, "permission": {"task": "deny"},
                                "agent": {"title": {"disable": True}}},
        }
    return req


async def _one(a, task_id, agent, out, idx):
    t0 = time.time()
    resp, server = await POOL.run(_request(a, task_id, agent), timeout_s=a.timeout)
    rec = {"idx": idx, "task_id": task_id, "agent": agent, "server": server, "wall_s": round(time.time() - t0, 1),
           "response": resp}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    status = (resp or {}).get("exit_status")
    print(f"[{idx}] {task_id} {agent} -> {status} reward={(resp or {}).get('reward')} {rec['wall_s']}s via {server}",
          flush=True)
    return rec


async def main_async(a: argparse.Namespace) -> None:
    POOL.refresh(force=True)
    print("agent servers:", POOL.snapshot(), flush=True)
    ids = _task_ids(a)
    if a.mode == "trials":
        jobs = [(t, ag) for ag in a.agents.split(",") for t in ids for _ in range(a.attempts)]
    elif a.mode == "nop":
        jobs = [(ids[i % len(ids)], "nop") for i in range(a.n)]
    elif a.mode == "sleepflush":  # sandboxes that just sleep; flush_all after --after s
        jobs = [(ids[i % len(ids)], "sleep") for i in range(a.n)]
    else:  # flush: start real trials, flush_all after --after s, verify they return
        jobs = [(ids[i % len(ids)], "opencode") for i in range(a.n)]
    tasks = [asyncio.create_task(_one(a, t, ag, a.out, i)) for i, (t, ag) in enumerate(jobs)]
    if a.mode in ("flush", "sleepflush"):
        await asyncio.sleep(a.after)
        print("flush_all ->", await POOL.broadcast("/flush_all", {}), flush=True)
    await asyncio.gather(*tasks)
    print("pool:", POOL.snapshot(), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["trials", "nop", "flush", "sleepflush"])
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--task-ids", default="")
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--agents", default="opencode", help="comma list of Harbor agent names (trials mode)")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--after", type=float, default=120)
    p.add_argument("--base-url", default="http://127.0.0.1:30600/v1")
    p.add_argument("--model", default="openai/qwen35-9b")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--max-seq-len", type=int, default=131072)
    p.add_argument("--agent-import-path", default="harbor_miles_agents.opencode_agents:BbhOpenCode")
    p.add_argument("--timeout", type=float, default=5400)
    p.add_argument("--out", required=True)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
