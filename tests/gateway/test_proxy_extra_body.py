"""extra_body (LoRA adapter selection) and session routing header on proxied requests."""

from __future__ import annotations

import asyncio
import json

import httpx

from polar.config.topology import GatewayNodeConfig
from polar.gateway.engine import SGLangEngine
from polar.gateway.proxy import InferenceClient


def _run_completion(client_kwargs: dict, routing_key: str | None) -> tuple[dict, dict]:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "x"}}]})

    async def run() -> None:
        client = InferenceClient("http://router:9000", SGLangEngine(training_sampling=True), **client_kwargs)
        client._client = httpx.AsyncClient(base_url="http://router:9000", transport=httpx.MockTransport(handler))
        try:
            await client.completion(
                {"messages": [{"role": "user", "content": "hi"}], "lora_path": "harness-choice", "temperature": 0.2},
                routing_key=routing_key,
            )
        finally:
            await client.close()

    asyncio.run(run())
    return seen["body"], seen["headers"]


def test_extra_body_overrides_request_and_routing_header_carries_session() -> None:
    body, headers = _run_completion(
        {"extra_body": {"lora_path": "miles_lora"}, "routing_key_header": "X-SMG-Routing-Key"}, "sess-1"
    )
    assert body["lora_path"] == "miles_lora"
    assert body["temperature"] == 1.0  # training_sampling still pins sampling
    assert body["return_meta_info"] is True
    assert headers["x-smg-routing-key"] == "sess-1"


def test_defaults_add_nothing() -> None:
    body, headers = _run_completion({}, "sess-1")
    assert body["lora_path"] == "harness-choice"
    assert "x-smg-routing-key" not in headers


def test_topology_node_exposes_inference_extras() -> None:
    node = GatewayNodeConfig.model_validate(
        {
            "id": "node-01",
            "public_url": "http://127.0.0.1:8100",
            "inference": {
                "base_url": "http://127.0.0.1:9000",
                "extra_body": {"lora_path": "miles_lora"},
                "routing_key_header": "X-SMG-Routing-Key",
            },
        }
    )
    assert node.extra_body == {"lora_path": "miles_lora"}
    assert node.routing_key_header == "X-SMG-Routing-Key"
