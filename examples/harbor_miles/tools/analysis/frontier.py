"""27B LoRA RL frontier model (MODEL, composed from MEASURED / REPLAY components).

step ~= max(rollout, train) + sync   (async pool, staleness <= 2)
  rollout = B * decode_per_session / (engines * R(running))        R = measured engine decode curve (tok/s vs running requests)
  train   = B * train_tokens_per_session / trainer_tok_s            trainer_tok_s from real-run / trace replay at the cap
  running = in_flight * (1 - tool_share)                            sessions in tool/verifier execution hold KV but do not decode
  in_flight per engine <= knee = min(KV_tokens / resident_ctx, mamba_slots)
B_min (per engine, to keep the engine at >= `util` of its peak with an async pool of 2 batches): in_flight_needed * engines / 2.

Inputs are a JSON spec (workload per cap, engine curve, trainers); see SPEC_SWEGYM below for the format.
usage: python frontier.py [SPEC.json]
  no arg: built-in SWE-Gym 27B spec (SPEC_SWEGYM, from the r2 / chunk-1 measurements)
  specs/de4-27b-provisional.json: 27B de4 spec (PROVISIONAL base-pass workload, 64k and 96k caps)
Prints, per cap and batch size B in (64, 128, 256), the fastest layout per node count (1, 2, 4, 8 nodes x 8 GPUs; merged
serving). Labels: MEASURED (real-run logs), REPLAY (trainer replay of real traces), STACK (stack-agent measurements),
MODEL (derived).
Copied from QWEN27B frontier.py + de4_spec_provisional.json (2026-10-01); only this docstring changed.
"""
import json
import sys

SPEC_SWEGYM = {
    "label": "27B SWE-Gym rand48 (codex)",
    # per cap: decode tokens/session, final total tokens/session (train tokens before inflation), tool-time share, truncation at the cap,
    # train-token inflation (Path B TITO ~1.0; Path A Polar patched ~1.0 measured), success (base), provenance
    "workload": {
        "64k": {"decode": 27.0e3, "total": 45.0e3, "tool_share": 0.12, "trunc": 0.14, "inflation": 1.0, "success": 0.55, "zv_drop": 0.40,
                "src": "MEASURED r2 (resp len 25-29k, total 42-50k, ~42 of 48 running, 29 of ~70 groups dropped)"},
    },
    # engine: TP4 bf16 (merged serving = plain engine); decode curve measured on real 27B sessions (~26k ctx/request), adapter serving
    "engine": {"gpus": 4, "kv_tokens": 1.89e6, "mamba_slots": 144, "merged_speedup": 1.0 / 0.77,
               "curve": [[14, 901], [18, 1027], [22, 1138], [26, 1278], [30, 1411], [34, 1463], [38, 1518], [42, 1575], [46, 1670]],
               "inflight_cap": 48,  # measured range ends at 48 in flight (sandboxes); flat beyond the last point (conservative)
               "src": "MEASURED (r2/chunk-1 SGLang logs, adapter serving); merged x1/0.77 = STACK"},
    # trainers per cap: gpus, tok/s, peak GB, provenance
    "trainers": {
        "64k": [{"name": "4 GPU TP4 CP1", "gpus": 4, "tok_s": 6.7e3, "peak_gb": 79, "src": "MEASURED r2 steps 2-4"},
                {"name": "8 GPU TP4 DP2", "gpus": 8, "tok_s": 14.5e3, "peak_gb": 79, "src": "REPLAY real traces"},
                {"name": "8 GPU TP4 CP2 headwise", "gpus": 8, "tok_s": 10.0e3, "peak_gb": 58, "src": "REPLAY real traces"}],
    },
    "sync_s": 15,
}


def R(curve, n, merged=1.0):
    """Engine decode tok/s at n running requests (linear interpolation, flat beyond the last point)."""
    pts = sorted(curve)
    if n <= pts[0][0]:
        return pts[0][1] * n / pts[0][0] * merged
    for (a, ra), (b, rb) in zip(pts, pts[1:]):
        if n <= b:
            return (ra + (rb - ra) * (n - a) / (b - a)) * merged
    return pts[-1][1] * merged


def layouts(spec, cap, B, nodes):
    """Enumerate split layouts on `nodes` x 8 GPUs: trainer option + remaining GPUs as TP4 engines."""
    eng = spec["engine"]; w = spec["workload"][cap]
    out = []
    for tr in spec["trainers"].get(cap, []):
        free = nodes * 8 - tr["gpus"]
        if free < eng["gpus"]:
            continue
        E = free // eng["gpus"]
        resident = w["total"] / 2                      # mean context held by an in-flight session over its life
        knee = min(eng["kv_tokens"] / resident, eng["mamba_slots"])
        gen = B / (1 - w.get("zv_drop", 0.0))          # sessions generated per trained step (zero-variance groups refilled)
        inflight = min(knee, 2 * gen / E, eng.get("inflight_cap", 1e9))   # async pool of 2 batches over E engines
        running = inflight * (1 - w["tool_share"])
        for serve, f in (("adapter", 1.0), ("merged", eng["merged_speedup"])):
            rate = E * R(eng["curve"], running, f)
            roll = gen * w["decode"] / rate
            train = B * w["total"] * w["inflation"] / tr["tok_s"]
            step = max(roll, train) + spec["sync_s"]
            out.append(dict(gen=gen, cap=cap, B=B, nodes=nodes, trainer=tr["name"], engines=E, inflight=round(inflight), serve=serve,
                            rollout_min=roll / 60, train_min=train / 60, step_min=step / 60, steps_h=60 / (step / 60),
                            steps_node_h=60 / (step / 60) / nodes, sessions_h=B * 3600 / step,
                            train_Mtok_h=B * w["total"] * w["inflation"] * 3600 / step / 1e6,
                            bound="rollout" if roll > train else "train"))
    return out


def main():
    spec = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else SPEC_SWEGYM
    print(f"# {spec['label']}  [MODEL from: workload {', '.join(v['src'] for v in spec['workload'].values())}; engine {spec['engine']['src']}]")
    for cap in spec["workload"]:
        for B in (64, 128, 256):
            rows = []
            for nodes in (1, 2, 4, 8):
                rows += layouts(spec, cap, B, nodes)
            rows = [r for r in rows if r["serve"] == "merged"]
            print(f"\n## cap {cap}, B={B} (merged serving)")
            print("| nodes | trainer | engines x TP4 | in-flight/eng | rollout min | train min | step min | steps/node-h | sessions/h | train Mtok/h | bound |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            best = {}
            for r in rows:
                k = r["nodes"]
                if k not in best or r["step_min"] < best[k]["step_min"]:
                    best[k] = r
            for k in sorted(best):
                r = best[k]
                print(f"| {k} | {r['trainer']} | {r['engines']} | {r['inflight']} | {r['rollout_min']:.1f} | {r['train_min']:.1f} | {r['step_min']:.1f} | "
                      f"{r['steps_node_h']:.2f} | {r['sessions_h']:.0f} | {r['train_Mtok_h']:.0f} | {r['bound']} |")


if __name__ == "__main__":
    main()
