"""Least-loaded dispatch of Harbor trials over one agent server per node.

Miles runs every agent-function call inside ONE process (the RolloutExecutor
actor), so in-process ``Trial.run()`` would start every sandbox on that one
node. Instead each job node runs a Harbor agent server (patched for the
singularity backend) that starts sandboxes on its own node, and this module
spreads ``/run`` calls across them. Because every call goes through this one
process, the in-flight count per server kept here is exact; no coordination
service is needed.

Server discovery: ``HARBOR_AGENT_SERVERS`` (comma-separated base URLs) or
``HARBOR_AGENT_SERVERS_FILE`` (one ``url capacity`` per line, appended by each
node's entry script as its server comes up; re-read every 30 s so late nodes
join). Capacity = the server's ``--max-concurrent``.

A server that refuses connections is benched for 60 s and the call moves to
another server (safe: the trial never started). Errors after the request was
accepted are NOT retried (the trial may have run) and surface as a failed
trial to the caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BENCH_SEC = 60.0
_REFRESH_SEC = 30.0


@dataclass
class _Server:
    url: str
    capacity: int
    inflight: int = 0
    benched_until: float = 0.0
    started: int = 0
    failures: int = 0

    @property
    def load(self) -> float:
        return self.inflight / max(1, self.capacity)


@dataclass
class AgentServerPool:
    servers: dict[str, _Server] = field(default_factory=dict)
    _last_refresh: float = 0.0
    _client: httpx.AsyncClient | None = None
    _cond: asyncio.Condition | None = None

    # ---------------------------------------------------------------- discovery
    def refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < _REFRESH_SEC and self.servers:
            return
        self._last_refresh = now
        entries: list[tuple[str, int]] = []
        default_cap = int(os.environ.get("HARBOR_AGENT_SERVER_CAPACITY", "32"))
        if raw := os.environ.get("HARBOR_AGENT_SERVERS"):
            entries += [(u.strip(), default_cap) for u in raw.split(",") if u.strip()]
        if path := os.environ.get("HARBOR_AGENT_SERVERS_FILE"):
            try:
                with open(path) as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            entries.append((parts[0], int(parts[1]) if len(parts) > 1 else default_cap))
            except FileNotFoundError:
                pass
        for url, cap in entries:
            url = url.rstrip("/")
            if url in self.servers:
                self.servers[url].capacity = cap
            else:
                self.servers[url] = _Server(url=url, capacity=cap)
                logger.info("harbor dispatch: added agent server %s (capacity %d)", url, cap)

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            socket_options = [
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPIDLE", 4), 60),
                (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPINTVL", 5), 30),
                (socket.IPPROTO_TCP, getattr(socket, "TCP_KEEPCNT", 6), 5),
            ]
            self._client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(socket_options=socket_options),
                limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
                timeout=httpx.Timeout(None, connect=15.0),
            )
        return self._client

    def cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    # ---------------------------------------------------------------- placement
    def _pick(self) -> _Server | None:
        now = time.monotonic()
        live = [s for s in self.servers.values() if s.benched_until <= now and s.inflight < s.capacity]
        if not live:
            return None
        best = min(s.load for s in live)
        return random.choice([s for s in live if s.load == best])

    async def acquire(self) -> _Server:
        """Wait for a server with a free slot (least loaded wins)."""
        async with self.cond():
            while True:
                self.refresh()
                server = self._pick()
                if server is not None:
                    server.inflight += 1
                    server.started += 1
                    return server
                if not self.servers:
                    self.refresh(force=True)
                    if not self.servers:
                        raise RuntimeError(
                            "no Harbor agent servers configured "
                            "(HARBOR_AGENT_SERVERS / HARBOR_AGENT_SERVERS_FILE)"
                        )
                try:
                    await asyncio.wait_for(self.cond().wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass

    async def release(self, server: _Server) -> None:
        async with self.cond():
            server.inflight -= 1
            self.cond().notify_all()

    # ---------------------------------------------------------------- calls
    async def run(self, request: dict[str, Any], timeout_s: float) -> tuple[dict[str, Any] | None, str]:
        """POST /run on the least-loaded server. Returns (response|None, server_url)."""
        attempts = 0
        while True:
            server = await self.acquire()
            try:
                resp = await asyncio.wait_for(
                    self.client().post(f"{server.url}/run", json=request), timeout=timeout_s
                )
                resp.raise_for_status()
                return resp.json(), server.url
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                server.failures += 1
                server.benched_until = time.monotonic() + _BENCH_SEC
                attempts += 1
                logger.warning("harbor dispatch: %s unreachable (%r); benched", server.url, exc)
                if attempts >= max(3, len(self.servers)):
                    return None, server.url
            except asyncio.TimeoutError:
                logger.error("harbor dispatch: /run on %s exceeded %.0fs", server.url, timeout_s)
                return None, server.url
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                server.failures += 1
                logger.error("harbor dispatch: /run on %s failed: %r", server.url, exc)
                return None, server.url
            finally:
                await self.release(server)

    async def broadcast(self, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        self.refresh(force=True)
        out: dict[str, Any] = {}

        async def one(url: str) -> None:
            try:
                r = await self.client().post(f"{url}{path}", json=payload, headers=headers, timeout=60.0)
                out[url] = r.json() if r.status_code == 200 else {"status": r.status_code}
            except Exception as exc:
                out[url] = {"error": repr(exc)}

        await asyncio.gather(*(one(u) for u in self.servers))
        return out

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            u: {"inflight": s.inflight, "capacity": s.capacity, "started": s.started, "failures": s.failures}
            for u, s in self.servers.items()
        }


POOL = AgentServerPool()
