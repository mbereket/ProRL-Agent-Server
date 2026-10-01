#!/usr/bin/env python3
"""Merge real rollout dumps (--save-debug-rollout-data files) into bigger replay steps.

    compose_replay.py OUT_DIR IN.pt [IN.pt ...] --steps 3 [--seed 0]

Every output step <k>.pt holds ALL input samples (groups kept contiguous, group order shuffled per step so
packing varies), re-indexed: unique index / group_index, and rollout_id (Miles' trajectory id) remapped per
(input file, original id) so trajectories from different dumps never collide. Prints the length stats.
"""
import argparse, os, random, statistics
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("inputs", nargs="+")
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    groups = []  # list of lists of sample dicts
    for fi, path in enumerate(a.inputs):
        S = torch.load(path, weights_only=False)["samples"]
        by_g = {}
        for s in S:
            by_g.setdefault(s.get("group_index"), []).append(s)
        for g, ss in by_g.items():
            for s in ss:
                s["_traj"] = (fi, s.get("rollout_id") if s.get("rollout_id") is not None else ("i", s.get("index")))
            groups.append(ss)
    sizes = sorted({len(g) for g in groups})
    L = [len(s["tokens"]) for g in groups for s in g]
    print(f"{len(a.inputs)} dumps -> {len(groups)} groups (sizes {sizes}), {len(L)} samples, {sum(L)} tokens; "
          f"len min/med/p90/max {min(L)}/{int(statistics.median(L))}/{sorted(L)[int(.9*len(L))]}/{max(L)}")
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    for k in range(a.steps):
        order = list(range(len(groups))); rng.shuffle(order)
        out, traj_ids, idx = [], {}, 0
        for gi, g in enumerate(order):
            for s in groups[g]:
                t = dict(s); key = t.pop("_traj")
                t["group_index"] = gi; t["index"] = idx; idx += 1
                t["rollout_id"] = traj_ids.setdefault(key, len(traj_ids))
                out.append(t)
        torch.save(dict(rollout_id=k, metadata={}, samples=out), os.path.join(a.out, f"{k}.pt"))
    print(f"wrote {a.steps} steps x {len(L)} samples to {a.out}")


if __name__ == "__main__":
    main()
