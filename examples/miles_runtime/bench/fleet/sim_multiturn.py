"""Agentic multi-turn rollout simulator against an SGLang server (/generate, token ids).

Each simulated agent starts from a prompt of PROMPT_LEN random tokens and repeats
{generate TURN_OUT tokens (ignore_eos); append TOOL_LEN random "tool output" tokens}
until the context reaches MAX_CTX. Every turn re-sends the full context, like an agent
CLI does, so the server's radix/prefix cache (and, for GDN hybrids, its mamba-state
cache) decides how much gets re-prefilled.

Reports aggregate decode tokens/s, "trajectory tokens"/s (all context tokens that
finished), per-turn latency, and cached-token ratio. Samples /metrics (if enabled)
for running-request counts and token usage.

Usage: python sim_multiturn.py --url http://127.0.0.1:30000 --agents 32 --max-ctx 65536 \
          --prompt-len 4096 --turn-out 384 --tool-len 1536 --out result.json [--lora NAME]
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import time

import aiohttp


async def scrape_metrics(session, url, samples, stop):
    pat = re.compile(r'^sglang:(num_running_reqs|num_queue_reqs|token_usage|cache_hit_rate|gen_throughput)\{[^}]*\}\s+([0-9.eE+-]+)', re.M)
    while not stop.is_set():
        try:
            async with session.get(f"{url}/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                text = await r.text()
            row = {"t": time.time()}
            for k, v in pat.findall(text):
                row[k] = row.get(k, 0.0) + float(v)
            samples.append(row)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass


async def agent(i, args, session, stats, vocab_hi):
    rng = random.Random(args.seed * 100003 + i)
    ctx = [rng.randrange(1000, vocab_hi) for _ in range(args.prompt_len)]
    turns = 0
    while len(ctx) + args.turn_out <= args.max_ctx:
        payload = {
            "input_ids": ctx,
            "sampling_params": {"max_new_tokens": args.turn_out, "temperature": 1.0, "top_p": 1.0, "ignore_eos": True},
        }
        if args.lora:
            payload["lora_path"] = args.lora
        if args.logprob:
            payload["return_logprob"] = True
            payload["logprob_start_len"] = -1
        t0 = time.time()
        async with session.post(f"{args.url}/generate", json=payload) as r:
            if r.status != 200:
                stats["errors"].append(f"{r.status}: {(await r.text())[:300]}")
                return
            out = await r.json()
        dt = time.time() - t0
        meta = out.get("meta_info", {})
        out_ids = out.get("output_ids") or []
        stats["turn_lat"].append(dt)
        stats["events"].append((i, turns, t0, t0 + dt, len(out_ids), len(ctx), meta.get("cached_tokens", 0)))
        stats["decoded"] += len(out_ids)
        stats["prompt_tokens"] += meta.get("prompt_tokens", len(ctx))
        stats["cached_tokens"] += meta.get("cached_tokens", 0)
        ctx = ctx + list(out_ids) + [rng.randrange(1000, vocab_hi) for _ in range(args.tool_len)]
        turns += 1
    stats["traj_tokens"] += len(ctx)
    stats["turns"].append(turns)
    stats["done"] += 1


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--agents", type=int, default=32)
    ap.add_argument("--max-ctx", type=int, default=65536)
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--turn-out", type=int, default=384)
    ap.add_argument("--tool-len", type=int, default=1536)
    ap.add_argument("--vocab-hi", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", default="")
    ap.add_argument("--logprob", action="store_true")
    ap.add_argument("--time-limit", type=float, default=1800)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = {"turn_lat": [], "decoded": 0, "prompt_tokens": 0, "cached_tokens": 0, "traj_tokens": 0,
             "turns": [], "done": 0, "errors": [], "events": []}
    samples = []
    stop = asyncio.Event()
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
        mt = asyncio.create_task(scrape_metrics(session, args.url, samples, stop))
        t0 = time.time()
        tasks = [asyncio.create_task(agent(i, args, session, stats, args.vocab_hi)) for i in range(args.agents)]
        done, pending = await asyncio.wait(tasks, timeout=args.time_limit)
        for p in pending:
            p.cancel()
        wall = time.time() - t0
        stop.set()
        await mt

    lat = sorted(stats["turn_lat"]) or [0]
    # steady window: every agent has finished its first (cold) turn and none has finished its trajectory
    ev = stats["events"]
    first_done = {}
    last_end = {}
    for a, k, ts, te, n, L, c in ev:
        if k == 0:
            first_done[a] = te
        last_end[a] = max(last_end.get(a, 0), te)
    steady = None
    if len(first_done) == args.agents:
        w0 = max(first_done.values())
        w1 = min(last_end.values()) if stats["done"] == args.agents else wall + t0
        if w1 > w0 + 10:
            # decoded tokens attributed pro rata to the window
            dec = 0.0
            ctxs = []
            for a, k, ts, te, n, L, c in ev:
                if k == 0 or te <= w0 or ts >= w1:
                    continue
                frac = (min(te, w1) - max(ts, w0)) / max(1e-6, te - ts)
                dec += n * frac
                ctxs.append(L)
            steady = {"window_s": w1 - w0, "decode_tok_s": dec / (w1 - w0),
                      "ctx_mean": statistics.mean(ctxs) if ctxs else None}
    running = [s.get("num_running_reqs", 0) for s in samples]
    usage = [s.get("token_usage", 0) for s in samples]
    res = {
        "label": args.label,
        "config": vars(args),
        "wall_s": wall,
        "agents_done": stats["done"],
        "timed_out": len(pending),
        "decode_tok_s": stats["decoded"] / wall,
        "traj_tok_s": stats["traj_tokens"] / wall if stats["done"] else None,
        "prompt_tok_s": stats["prompt_tokens"] / wall,
        "cached_ratio": stats["cached_tokens"] / max(1, stats["prompt_tokens"]),
        "turns_mean": statistics.mean(stats["turns"]) if stats["turns"] else 0,
        "turn_lat_p50": lat[len(lat) // 2],
        "turn_lat_p90": lat[int(len(lat) * 0.9)],
        "running_reqs_mean": statistics.mean(running) if running else None,
        "running_reqs_max": max(running) if running else None,
        "token_usage_max": max(usage) if usage else None,
        "steady": steady,
        "errors": stats["errors"][:5],
        "n_errors": len(stats["errors"]),
    }
    json.dump({"summary": res, "metrics": samples, "events": [(a, k, ts - t0, te - t0, n, L, c) for a, k, ts, te, n, L, c in ev]}, open(args.out, "w"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
