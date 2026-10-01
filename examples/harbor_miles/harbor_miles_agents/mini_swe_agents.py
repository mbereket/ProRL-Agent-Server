"""Harbor mini-swe-agent with a host-built, read-only toolchain (setup/ensure_agent_tools.sh).

Harbor's MiniSweAgent installs build tools (apt), uv and mini-swe-agent into every sandbox at
trial time. Here the same mini-swe-agent (with Harbor's output-cap and step-timing patches
applied at build time) lives in HM_AGENT_TOOLS, bind-mounted into the sandbox at the same path:
install() only checks it, and every agent exec puts its bin/ first on PATH. Everything else —
model wiring, max_seq_len enforcement, trajectory parsing — is Harbor's MiniSweAgent unchanged.
"""

from __future__ import annotations

import os
from typing import Any

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment


def _tools_bin() -> str:
    root = os.environ.get("HM_AGENT_TOOLS_MINI_SWE_AGENT")
    if not root:
        raise RuntimeError("HM_AGENT_TOOLS_MINI_SWE_AGENT is not set (setup/ensure_agent_tools.sh)")
    return os.path.join(root, "bin")


class PreinstalledMiniSweAgent(MiniSweAgent):
    async def install(self, environment: BaseEnvironment) -> None:
        result = await self.exec_as_agent(environment, command="mini-swe-agent --help >/dev/null && echo ok")
        if getattr(result, "return_code", 0):
            raise RuntimeError(
                f"preinstalled mini-swe-agent not usable from {_tools_bin()}:\n{getattr(result, 'stdout', '')}"
            )

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        command = f'export PATH="{_tools_bin()}:$PATH"; {command}'
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)
