"""How does an agent harness replay its history? (TITO session v1/v2 + matcher decision)

    python replay_check.py calls.jsonl            # from tools/log_proxy.py

For each chat request, find the session predecessor (the earlier request whose messages are the
longest prefix of this one, compared up to the previous assistant turn) and classify how the
harness replayed the model's previous response:

  strict           replayed assistant message equal to the model's response (role/content/
                   reasoning_content/tool_calls)  -> v1 extends with --session-message-matcher strict
  loose_tool_call  equal except tool-call arguments re-serialized (same JSON)  -> needs loose_tool_call
  reasoning_dropped / content_changed / tool_calls_changed
                   -> v1 rejects/rolls back; v2 would branch (separate sample)
  root             first request of a session (no predecessor)
  non_prefix       not an extension of any earlier request (history edited/compacted, or a side call
                   such as title generation)  -> v1 starts a new session or rejects

Also reports system-prompt/tool-list changes between turns and the side-call share.
"""

from __future__ import annotations

import collections
import json
import sys


def _key(m: dict) -> str:
    return json.dumps(m, sort_keys=True)


def _norm_tc(tcs):
    out = []
    for tc in tcs or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = fn.get("arguments")
        out.append((tc.get("id"), fn.get("name"), json.dumps(args, sort_keys=True)))
    return out


def classify(resp: dict, replay: dict) -> str:
    rc, pc = resp.get("content") or "", replay.get("content") or ""
    rr, pr = resp.get("reasoning_content") or "", replay.get("reasoning_content") or ""
    rt, pt = resp.get("tool_calls") or [], replay.get("tool_calls") or []
    if rr and not pr:
        return "reasoning_dropped"
    if rr != pr:
        return "reasoning_changed"
    if rc.strip() != pc.strip():
        return "content_changed"
    if len(rt) != len(pt):
        return "tool_calls_changed"
    raw_equal = all((a.get("function") or {}).get("arguments") == (b.get("function") or {}).get("arguments")
                    and a.get("id") == b.get("id") for a, b in zip(rt, pt))
    if raw_equal and rc == pc:
        return "strict"
    if _norm_tc(rt) == _norm_tc(pt):
        return "loose_tool_call"
    return "tool_calls_changed"


def main(path: str) -> None:
    calls = [json.loads(line) for line in open(path) if line.strip()]
    reqs = [(c.get("request") or {}).get("messages") or [] for c in calls]
    keys = [[_key(m) for m in r] for r in reqs]
    counts = collections.Counter()
    examples = {}
    tools_changed = 0
    for j, mj in enumerate(reqs):
        # Longest-prefix predecessor. Sibling sessions (several attempts of one task) share identical
        # early requests, so among equally long candidates prefer the one whose response this
        # request actually replays (the same session); fall back to the latest.
        cands = [i for i in range(j) if len(keys[i]) < len(keys[j]) and keys[j][:len(keys[i])] == keys[i]]
        best = None
        if cands:
            longest = max(len(keys[i]) for i in cands)
            same_len = [i for i in cands if len(keys[i]) == longest]
            nxt = mj[longest] if longest < len(mj) else {}
            matching = [i for i in same_len if nxt.get("role") == "assistant"
                        and classify(calls[i].get("response") or {}, nxt) in ("strict", "loose_tool_call")]
            best = (matching or same_len)[-1]
        if best is None:
            # root, or a request that does not extend any earlier one
            is_root = len(mj) <= 2 or all(m.get("role") in ("system", "user") for m in mj)
            label = "root" if is_root else "non_prefix"
        else:
            nxt = mj[len(reqs[best])]
            if nxt.get("role") != "assistant":
                label = "non_assistant_append"
            else:
                label = classify(calls[best].get("response") or {}, nxt)
            if (calls[best].get("request") or {}).get("tools") != (calls[j].get("request") or {}).get("tools"):
                tools_changed += 1
        counts[label] += 1
        examples.setdefault(label, j)
    n = len(calls)
    print(f"{n} requests")
    for k, v in counts.most_common():
        print(f"  {k:22s} {v:5d}  ({v / n:.1%})")
    print(f"  tool list changed between turns: {tools_changed}")
    v1_ok = counts["strict"] + counts["loose_tool_call"]
    turns = n - counts["root"]
    if turns:
        print(f"v1 extendable turns: strict {counts['strict'] / turns:.1%}, with loose_tool_call {v1_ok / turns:.1%}")


if __name__ == "__main__":
    main(sys.argv[1])
