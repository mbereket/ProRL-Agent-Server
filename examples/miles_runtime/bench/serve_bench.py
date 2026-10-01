#!/usr/bin/env python3
"""Decode-throughput client for SGLang servers (fixed-length forced decode, ignore_eos).

    serve_bench.py --url http://127.0.0.1:PORT --name NAME --out DIR [--lora miles_lora]
                   [--shapes 2048x2048x64,32768x1024x16]   # prompt_tokens x gen_tokens x concurrency

Every shape sends 2*concurrency requests (closed loop, `concurrency` in flight) of random-token prompts
(input_ids, so prompt length is exact) and records output tokens / wall time. One JSON line per shape
is appended to OUT/serve.jsonl.
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp


async def one(session, url, ids, gen, lora):
    payload = {"input_ids": ids, "sampling_params": {"max_new_tokens": gen, "ignore_eos": True, "temperature": 1.0}}
    if lora:
        payload["lora_path"] = lora
    async with session.post(f"{url}/generate", json=payload) as r:
        r.raise_for_status()
        out = await r.json()
    return out["meta_info"]["completion_tokens"]


async def run_shape(url, lora, plen, gen, conc, seed):
    rng = random.Random(seed)
    n = 2 * conc
    prompts = [[rng.randrange(1000, 150000) for _ in range(plen)] for _ in range(n)]
    sem = asyncio.Semaphore(conc)
    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        await one(s, url, prompts[0][:64], 8, lora)  # warmup (and LoRA load)

        async def task(p):
            async with sem:
                return await one(s, url, p, gen, lora)

        t0 = time.time()
        toks = await asyncio.gather(*(task(p) for p in prompts))
        dt = time.time() - t0
    return {"prompt": plen, "gen": gen, "conc": conc, "requests": n, "out_tokens": sum(toks),
            "wall_s": dt, "out_tok_per_s": sum(toks) / dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--shapes", default="2048x2048x64,32768x1024x16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for i, sh in enumerate(a.shapes.split(",")):
        plen, gen, conc = (int(x) for x in sh.split("x"))
        res = asyncio.run(run_shape(a.url, a.lora, plen, gen, conc, seed=i))
        res["name"] = a.name
        print(json.dumps(res), flush=True)
        with open(os.path.join(a.out, "serve.jsonl"), "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
