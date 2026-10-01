#!/usr/bin/env python3
"""Merge real rollout dumps (--save-debug-rollout-data files) into bigger replay steps.

    compose_replay.py OUT_DIR IN.pt [IN.pt ...] --steps 3 [--seed 0] [--max-samples N]

Every output step <k>.pt holds ALL input samples (groups kept contiguous, group order shuffled per step so
packing varies), re-indexed: unique index / group_index, and rollout_id (Miles' trajectory id) remapped per
(input file, original id) so trajectories from different dumps never collide. Prints the length stats.
--max-samples N keeps N samples per step (whole groups, seeded random order per step, the group holding the
longest sample always first so every step exercises the peak length); set RBS * NS = N for the replay arm.
"""
import argparse, os, pickle, random, statistics, types
import torch


class _Stub:  # stands in for classes from packages absent in the runtime (polar, slime_bridge, ...)
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): self.__dict__["_state"] = state


class _TolerantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _Stub


_pickle_shim = types.SimpleNamespace(Unpickler=_TolerantUnpickler, load=pickle.load, loads=pickle.loads,
                                     dump=pickle.dump, dumps=pickle.dumps, Pickler=pickle.Pickler,
                                     HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL, __name__="pickle")


def _load(path):
    return torch.load(path, weights_only=False, pickle_module=_pickle_shim)


def _clean(obj):
    """Drop anything that only unpickles with packages missing from the runtime."""
    if isinstance(obj, _Stub):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not isinstance(v, _Stub)}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clean(v) for v in obj)
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("inputs", nargs="+")
    ap.add_argument("--steps", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    a = ap.parse_args()
    groups = []  # list of lists of sample dicts
    for fi, path in enumerate(a.inputs):
        S = [_clean(x) for x in _load(path)["samples"]]
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
    longest = max(range(len(groups)), key=lambda g: max(len(s["tokens"]) for s in groups[g]))
    for k in range(a.steps):
        order = list(range(len(groups))); rng.shuffle(order)
        if a.max_samples:
            order.remove(longest); order.insert(0, longest)
        out, traj_ids, idx = [], {}, 0
        for gi, g in enumerate(order):
            for s in groups[g]:
                if a.max_samples and idx >= a.max_samples:
                    break
                t = dict(s); key = t.pop("_traj")
                t["metadata"] = {}  # bridge/Polar metadata is not needed to train
                t["group_index"] = gi; t["index"] = idx; idx += 1
                t["rollout_id"] = traj_ids.setdefault(key, len(traj_ids))
                out.append(t)
        torch.save(dict(rollout_id=k, metadata={}, samples=out), os.path.join(a.out, f"{k}.pt"))
        LS = [len(t["tokens"]) for t in out]
        print(f"step {k}: {len(out)} samples, {sum(LS)} tokens, max {max(LS)}")
    print(f"wrote {a.steps} steps to {a.out}")


if __name__ == "__main__":
    main()
