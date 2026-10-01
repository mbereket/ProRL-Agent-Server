#!/usr/bin/env python3
"""Write fixed-length synthetic rollout dumps for length-controlled training benchmarks.

    make_synthetic.py OUT_DIR --seq-len 16384 --samples 64 --group 8 --rollouts 4 [--prompt-frac 0.1]

Every sample has exactly --seq-len tokens (prompt + response), a full loss mask on the
response, dummy rollout logprobs and group-balanced 0/1 rewards (non-zero GRPO advantages),
so full-FT and LoRA arms train on byte-identical batches. Files: OUT_DIR/<rollout_id>.pt in
the --load-debug-rollout-data format.
"""
import argparse
import os
import random

import torch
from miles.utils.types import Sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--prompt-frac", type=float, default=0.1)
    ap.add_argument("--vocab-lo", type=int, default=1000)
    ap.add_argument("--vocab-hi", type=int, default=150000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(0)
    p_len = max(16, int(a.seq_len * a.prompt_frac))
    r_len = a.seq_len - p_len
    for rid in range(a.rollouts):
        samples = []
        for g in range(a.samples // a.group):
            prompt = [rng.randrange(a.vocab_lo, a.vocab_hi) for _ in range(p_len)]
            for j in range(a.group):
                idx = g * a.group + j
                resp = [rng.randrange(a.vocab_lo, a.vocab_hi) for _ in range(r_len)]
                samples.append(Sample(
                    # rollout_id is Miles' *trajectory* id (samples sharing it are segments of one
                    # trajectory and must share one reward) -- leave it unset: one sample per trajectory.
                    group_index=g, index=idx, prompt="synthetic", tokens=prompt + resp,
                    response="", response_length=r_len, label="", reward=float(j % 2),
                    loss_mask=[1] * r_len, rollout_log_probs=[-2.0] * r_len,
                    status=Sample.Status.COMPLETED,
                ).to_dict())
        torch.save(dict(rollout_id=rid, metadata={}, samples=samples), os.path.join(a.out, f"{rid}.pt"))
        print(f"wrote {a.out}/{rid}.pt: {len(samples)} x {a.seq_len} tokens")


if __name__ == "__main__":
    main()
