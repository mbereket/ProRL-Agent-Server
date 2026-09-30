"""Harbor OpenCode agents for training on our task images (loaded by import path).

The Harbor agent server runs these via ``agent_import_path`` in the /run request
(patch 0002); the connection config (provider baseURL -> Miles session URL)
still comes from ``agent_name = "opencode"``.

PreinstalledOpenCode
    Uses the opencode already in the task image (or on a read-only toolchain
    mount) instead of Harbor's per-trial apt + nvm + ``npm i -g opencode-ai``,
    which costs minutes and network at every sandbox start. Every exec gets a
    generous timeout: without one the singularity backend's HTTP client gives
    up on a long ``opencode run`` after 600 s.

BbhOpenCode
    BixBench-Hypothesis MCP tasks. The judge-backed MCP server is started from
    the agent setup step (the task env, including RUBRIC_MODEL_API_KEY, is
    applied to every exec but not to container bootstrap), and the episode
    ends at the first ``submit_answer``: the MCP server grades it, writes
    ``.harbor/artifacts/reward.json`` and refuses every later call, but nothing
    stops opencode, which would otherwise keep calling tools until its context
    or the agent timeout runs out. ``opencode run`` is wrapped in a watcher
    that stops it once reward.json exists. (Same contract as eval-v2's
    preinstalled_opencode_bbh.py, so train and eval episodes end the same way.)
"""

from __future__ import annotations

from typing import Any

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment

_NVM = ". ~/.nvm/nvm.sh >/dev/null 2>&1 || true; "


class PreinstalledOpenCode(OpenCode):
    def __init__(self, *args: Any, exec_timeout_sec: int = 7200, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._exec_timeout_sec = int(exec_timeout_sec)

    async def install(self, environment: BaseEnvironment) -> None:
        result = await self.exec_as_agent(
            environment,
            command=_NVM + "node --version && opencode --version",
            timeout_sec=120,
        )
        if getattr(result, "return_code", 0):
            raise RuntimeError(
                "preinstalled opencode not found in the sandbox:\n"
                f"{getattr(result, 'stdout', '') or ''}"
            )

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        return await super().exec_as_agent(
            environment,
            command,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec or self._exec_timeout_sec,
        )


# Runs the wrapped command in its own process group, polls for the reward file
# and kills the group once it appears. A trial that ends on its own keeps
# opencode's exit status; a trial stopped here exits 0.
STOP_AFTER_SUBMIT = """\
set -m
reward_file="${{WORKDIR:-/app}}/.harbor/artifacts/reward.json"
( {command} ) &
pid=$!
stopped=
while kill -0 "$pid" 2>/dev/null; do
  if [ -f "$reward_file" ]; then
    sleep {grace_sec}
    kill -TERM -- -"$pid" 2>/dev/null
    sleep 5
    kill -KILL -- -"$pid" 2>/dev/null
    stopped=1
    break
  fi
  sleep {poll_sec}
done
wait "$pid" 2>/dev/null
rc=$?
if [ -n "$stopped" ]; then echo "[stop-after-submit] reward.json present; opencode terminated"; exit 0; fi
exit $rc
"""


class BbhOpenCode(PreinstalledOpenCode):
    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if "opencode --model=" in command and " run " in command:
            command = STOP_AFTER_SUBMIT.format(command=command, grace_sec=5, poll_sec=2)
        return await super().exec_as_agent(
            environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        result = await self.exec_as_agent(
            environment,
            command="bash /staging/env_files/start_mcp.sh",
            timeout_sec=900,
        )
        if getattr(result, "return_code", 0):
            raise RuntimeError(
                "BixBench-Hypothesis MCP server failed to start:\n"
                f"{getattr(result, 'stdout', '') or ''}"
            )
