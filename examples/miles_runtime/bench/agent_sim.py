#!/usr/bin/env python3
"""Agentic-rollout load simulator for SGLang engines (prefix-cache-heavy multi-turn sessions).

    agent_sim.py --urls http://h:p,... --sessions 48 --turns 12 --name X --out DIR
                 [--base 6000] [--tool 1400] [--gen 600] [--ctx-cap 65536]

Each session: turn t sends input_ids = system/task prefix (`base` tokens) + all previous (reply + tool output)
pairs, then generates `gen` tokens (ignore_eos). Next turn appends the reply and `tool` new random tokens (the
"tool output"), so every request re-sends a growing prefix — the shape of codex/opencode sessions with router
session affinity (sessions pinned to one engine round-robin). Stops at `turns` or the context cap.
Reports generated tok/s, turn latency p50/p90, prefill tokens computed vs cached, completed sessions/min.
"""
import argparse, asyncio, json, os, random, statistics, time

import aiohttp


async def session(s, url, sid, a, stats):
    rng = random.Random(sid)
    ctx = [rng.randrange(1000, 150000) for _ in range(a.base)]
    for t in range(a.turns):
        if len(ctx) + a.gen > a.ctx_cap:
            break
        payload = {"input_ids": ctx, "sampling_params": {"max_new_tokens": a.gen, "ignore_eos": True, "temperature": 1.0}}
        t0 = time.time()
        async with s.post(f"{url}/generate", json=payload) as r:
            r.raise_for_status()
            out = await r.json()
        dt = time.time() - t0
        mi = out["meta_info"]
        stats["lat"].append(dt)
        stats["gen"] += mi["completion_tokens"]
        stats["prompt"] += mi.get("prompt_tokens", len(ctx))
        stats["cached"] += mi.get("cached_tokens", 0)
        ctx = ctx + out.get("output_ids", [0] * mi["completion_tokens"]) + [rng.randrange(1000, 150000) for _ in range(a.tool)]
        await asyncio.sleep(rng.uniform(0.0, a.tool_wait))  # tool execution time (sandbox)
    stats["done"] += 1


async def main_async(a):
    urls = a.urls.split(",")
    stats = {"lat": [], "gen": 0, "prompt": 0, "cached": 0, "done": 0}
    timeout = aiohttp.ClientTimeout(total=None)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        for u in urls:  # warmup
            async with s.post(f"{u}/generate", json={"input_ids": [1000] * 32, "sampling_params": {"max_new_tokens": 4}}) as r:
                await r.read()
        t0 = time.time()
        await asyncio.gather(*(session(s, urls[i % len(urls)], i, a, stats) for i in range(a.sessions)))
        wall = time.time() - t0
    lat = sorted(stats["lat"])
    res = {"name": a.name, "engines": len(urls), "sessions": a.sessions, "turns": a.turns, "wall_s": wall,
           "gen_tok_per_s": stats["gen"] / wall, "turns_per_s": len(lat) / wall,
           "sessions_per_min": stats["done"] / wall * 60, "lat_p50": lat[len(lat) // 2], "lat_p90": lat[int(.9 * len(lat))],
           "prefill_new_tok_per_s": (stats["prompt"] - stats["cached"]) / wall,
           "cache_hit": stats["cached"] / max(1, stats["prompt"])}
    print(json.dumps(res), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "agent_sim.jsonl"), "a") as f:
        f.write(json.dumps(res) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=48); ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--base", type=int, default=6000); ap.add_argument("--tool", type=int, default=1400)
    ap.add_argument("--gen", type=int, default=600); ap.add_argument("--ctx-cap", type=int, default=65536)
    ap.add_argument("--tool-wait", type=float, default=0.0, help="max uniform sleep between turns (s)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
