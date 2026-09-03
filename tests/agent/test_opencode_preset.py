from __future__ import annotations

import asyncio
import json
from pathlib import Path

from polar.agent.models import AgentSpec
from polar.agent.presets.opencode import OpenCodeHarness
from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec


class RecordingRuntime(BaseRuntime):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(RuntimeSpec(image="fake"), "session-1", tmp_path / "session")
        self.commands: list[str] = []

    @property
    def runtime_id(self) -> str:
        return "fake"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def exec(self, command: str, *, cwd=None, env=None, timeout_sec=None) -> ExecResult:
        self.commands.append(command)
        return ExecResult(stdout="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None: ...

    async def upload_dir(self, local_path: str, remote_path: str) -> None: ...

    async def download_file(self, remote_path: str, local_path: str) -> None: ...

    async def download_dir(self, remote_path: str, local_path: str) -> None: ...


def _written_config(runtime: RecordingRuntime) -> dict:
    heredoc = next(c for c in runtime.commands if "opencode.json" in c)
    body = heredoc.split("<< 'POLARCFG'\n", 1)[1].rsplit("\nPOLARCFG", 1)[0]
    return json.loads(body)


def test_opencode_disables_title_generation_by_default(tmp_path: Path) -> None:
    harness = OpenCodeHarness(AgentSpec(harness="opencode", model_name="openai/gpt-5.4"))
    runtime = RecordingRuntime(tmp_path)

    asyncio.run(harness.setup(runtime))

    config = _written_config(runtime)
    assert config["agent"]["title"]["disable"] is True
    assert config["provider"] == {"openai": {"models": {"gpt-5.4": {}}}}


def test_opencode_config_passthrough_deep_merges(tmp_path: Path) -> None:
    spec = AgentSpec(
        harness="opencode",
        model_name="openai/gpt-5.4",
        settings={
            "opencode_config": {
                "agent": {"title": {"disable": False}, "build": {"prompt": "custom"}},
                "compaction": {"auto": False},
            }
        },
    )
    runtime = RecordingRuntime(tmp_path)

    asyncio.run(OpenCodeHarness(spec).setup(runtime))

    config = _written_config(runtime)
    assert config["agent"]["title"]["disable"] is False
    assert config["agent"]["build"]["prompt"] == "custom"
    assert config["compaction"] == {"auto": False}
    assert config["permission"]["bash"] == "allow"
