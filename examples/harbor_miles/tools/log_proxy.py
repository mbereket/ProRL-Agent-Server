"""OpenAI-compatible pass-through proxy that records every chat request/response.

Used by the Harbor validation job to see exactly what an agent harness replays
on each turn (does it resend reasoning_content? re-serialize tool-call
arguments?), which decides the Miles session-server matcher / version.

    python log_proxy.py --upstream http://127.0.0.1:30500 --port 30600 --out calls.jsonl

Streaming responses are passed through chunk by chunk and also reassembled
(content, reasoning_content, tool_calls, finish_reason) for the log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()
_args: argparse.Namespace
_client: httpx.AsyncClient
_lock = asyncio.Lock()


async def _record(entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False)
    async with _lock:
        with open(_args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _merge_delta(acc: dict[str, Any], choice: dict[str, Any]) -> None:
    delta = choice.get("delta") or {}
    for key in ("content", "reasoning_content"):
        if delta.get(key):
            acc[key] = acc.get(key, "") + delta[key]
    for tc in delta.get("tool_calls") or []:
        idx = tc.get("index", 0)
        calls = acc.setdefault("tool_calls", {})
        cur = calls.setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
        if tc.get("id"):
            cur["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            cur["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            cur["function"]["arguments"] += fn["arguments"]
    if choice.get("finish_reason"):
        acc["finish_reason"] = choice["finish_reason"]


@app.api_route("/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    url = f"{_args.upstream.rstrip('/')}/{path}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    if request.method == "GET":
        r = await _client.get(url, headers=headers)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        payload = None
    call_id = uuid.uuid4().hex[:12]
    t0 = time.time()
    entry: dict[str, Any] = {"id": call_id, "t": t0, "path": path, "request": payload}
    if isinstance(payload, dict) and payload.get("stream"):
        upstream = _client.build_request("POST", url, content=body, headers=headers)
        resp = await _client.send(upstream, stream=True)
        acc: dict[str, Any] = {}

        async def gen():
            buf = b""
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        for choice in obj.get("choices") or []:
                            _merge_delta(acc, choice)
                        if obj.get("usage"):
                            acc["usage"] = obj["usage"]
            finally:
                await resp.aclose()
                if "tool_calls" in acc:
                    acc["tool_calls"] = [acc["tool_calls"][k] for k in sorted(acc["tool_calls"])]
                entry.update(status=resp.status_code, response=acc, dt=time.time() - t0)
                await _record(entry)

        return StreamingResponse(gen(), status_code=resp.status_code, media_type=resp.headers.get("content-type"))
    r = await _client.post(url, content=body, headers=headers)
    try:
        rj = r.json()
    except Exception:
        rj = {"raw": r.text[:2000]}
    msg = ((rj.get("choices") or [{}])[0].get("message")) if isinstance(rj, dict) else None
    entry.update(status=r.status_code, response=msg or rj, usage=(rj or {}).get("usage") if isinstance(rj, dict) else None,
                 dt=time.time() - t0)
    await _record(entry)
    return JSONResponse(content=rj, status_code=r.status_code)


def main() -> None:
    global _args, _client
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=30600)
    p.add_argument("--out", required=True)
    _args = p.parse_args()
    _client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=30.0))
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="warning")


if __name__ == "__main__":
    main()
