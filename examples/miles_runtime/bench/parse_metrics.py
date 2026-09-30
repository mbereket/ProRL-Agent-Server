#!/usr/bin/env python3
"""Turn a bench run dir (train.log, rollout_data/*.pt, gpu_mem.csv) into metrics.jsonl + summary.json.

metrics.jsonl: one row per rollout id merging Miles' `perf N: {...}`, `step N: {...}`,
`rollout N: {...}` console dicts, plus exact token counts from the dumped rollout data
(the length normalization used by the report: per-token train / decode costs).
"""
from __future__ import annotations

import ast
import csv
import glob
import json
import os
import re
import statistics
import sys

LINE = re.compile(r"\b(perf|step|rollout|eval) (\d+): (\{.*\})\s*$")
NUMWRAP = re.compile(r"(?:np\.(?:float|int)\d*|tensor|torch\.tensor)\(([^(),]+)(?:,[^()]*)?\)")


def parse_dict(text: str) -> dict:
    text = NUMWRAP.sub(r"\1", text)
    text = re.sub(r"([:,\[]\s*)-?(?:nan|inf)\b", r"\1None", text)
    try:
        d = ast.literal_eval(text)
    except Exception:
        return {}
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and v is not None}


def main(out: str) -> None:
    rows: dict[int, dict] = {}
    with open(os.path.join(out, "train.log"), errors="replace") as f:
        for line in f:
            m = LINE.search(line)
            if not m:
                continue
            kind, rid, body = m.group(1), int(m.group(2)), m.group(3)
            d = parse_dict(body)
            row = rows.setdefault(rid, {"rollout_id": rid})
            for k, v in d.items():
                row[k if "/" in k else f"{kind}/{k}"] = v

    for path in sorted(glob.glob(os.path.join(out, "rollout_data", "*.pt"))):
        stem = os.path.basename(path)[:-3]
        if not stem.isdigit():
            continue
        import torch

        payload = torch.load(path, weights_only=False)
        samples = payload["samples"]
        resp = [s.get("response_length", 0) for s in samples]
        tot = [len(s.get("tokens", [])) for s in samples]
        trunc = sum(1 for s in samples if str(s.get("status", "")).lower().endswith("truncated"))
        row = rows.setdefault(int(stem), {"rollout_id": int(stem)})
        row.update({
            "tok/n_samples": len(samples),
            "tok/response_total": sum(resp),
            "tok/sequence_total": sum(tot),
            "tok/response_mean": statistics.mean(resp) if resp else 0,
            "tok/response_max": max(resp) if resp else 0,
            "tok/truncated_frac": trunc / max(1, len(samples)),
        })

    for row in rows.values():
        rt, tt = row.get("perf/rollout_time"), row.get("perf/actor_train_time")
        if rt and row.get("tok/response_total"):
            row["norm/decode_tok_per_s"] = row["tok/response_total"] / rt
            row["norm/rollout_ms_per_resp_tok"] = 1000 * rt / row["tok/response_total"]
        if tt and row.get("tok/sequence_total"):
            row["norm/train_ms_per_tok"] = 1000 * tt / row["tok/sequence_total"]
        for key in ("fwd_bwd", "optimizer", "log_probs"):
            t = row.get(f"perf/{key}_time")
            if t and row.get("tok/sequence_total"):
                row[f"norm/{key}_ms_per_tok"] = 1000 * t / row["tok/sequence_total"]

    ordered = [rows[k] for k in sorted(rows)]
    with open(os.path.join(out, "metrics.jsonl"), "w") as f:
        for r in ordered:
            f.write(json.dumps(r) + "\n")

    peak: dict[str, float] = {}
    gpu_csv = os.path.join(out, "gpu_mem.csv")
    if os.path.exists(gpu_csv):
        with open(gpu_csv) as f:
            for rec in csv.reader(f):
                if len(rec) >= 3:
                    try:
                        peak[rec[1].strip()] = max(peak.get(rec[1].strip(), 0.0), float(rec[2]))
                    except ValueError:
                        pass

    def med(key: str, skip_first: bool = True):
        vals = [r[key] for r in ordered[1 if skip_first and len(ordered) > 1 else 0:] if key in r]
        return statistics.median(vals) if vals else None

    keys = sorted({k for r in ordered for k in r if k.startswith(("perf/", "norm/", "tok/")) or "abs_diff" in k
                   or k.endswith(("/raw_reward", "/rewards", "reward", "/loss", "/pg_loss", "/grad_norm", "/kl"))})
    summary = {
        "run": os.path.basename(out.rstrip("/")),
        "steps": len(ordered),
        "median_excl_first": {k: med(k) for k in keys},
        "first_step": {k: ordered[0].get(k) for k in keys} if ordered else {},
        "last_step": {k: ordered[-1].get(k) for k in keys} if ordered else {},
        "gpu_mem_peak_mib": peak,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in summary["median_excl_first"].items()
                      if k.startswith(("perf/rollout_time", "perf/actor_train", "perf/update_weights", "perf/step_time",
                                       "norm/", "tok/response_mean"))}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
