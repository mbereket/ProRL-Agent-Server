"""Harbor codex with a host-built, read-only toolchain (setup/ensure_agent_tools.sh codex).

Harbor's Codex agent installs Node + @openai/codex in every sandbox at trial time (network, slow,
impossible where sandboxes have no DNS). Here the pinned codex lives in HM_AGENT_TOOLS_CODEX,
bind-mounted read-only at the same path; install() only checks it and every exec puts its bin/
first on PATH. Everything else (auth file, config.toml base URL, `codex exec --json`, trajectory
parsing) is Harbor's Codex unchanged.

Extra codex config (agent kwarg ``codex_config``, a flat dict of dotted keys) is passed as ``-c
key=value`` flags. Training defaults (overridable):
  model_context_window           = HARBOR_MAX_SEQ_LEN  (codex does not know a custom model's window)
  model_auto_compact_token_limit = 1e9                 (auto-compaction rewrites history, which a
                                                        TITO session cannot extend; a run that
                                                        reaches the window ends as overlong instead)
"""

from __future__ import annotations

import os
import shlex
from typing import Any

from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment


def _tools_bin() -> str:
    root = os.environ.get("HM_AGENT_TOOLS_CODEX")
    if not root:
        raise RuntimeError("HM_AGENT_TOOLS_CODEX is not set (setup/ensure_agent_tools.sh codex)")
    return os.path.join(root, "bin")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace('"', '\\"') + '"'


class PreinstalledCodex(Codex):
    def __init__(self, *args: Any, codex_config: dict[str, Any] | None = None,
                 exec_timeout_sec: int = 7200, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        defaults: dict[str, Any] = {"model_auto_compact_token_limit": 1_000_000_000}
        if os.environ.get("HARBOR_MAX_SEQ_LEN"):
            defaults["model_context_window"] = int(os.environ["HARBOR_MAX_SEQ_LEN"])
        self._codex_config = {**defaults, **(codex_config or {})}
        self._exec_timeout_sec = int(exec_timeout_sec)

    def _get_env(self, key: str) -> str | None:
        # The agent server exports the session URL as OPENAI_API_BASE; codex reads OPENAI_BASE_URL.
        value = super()._get_env(key)
        if value is None and key == "OPENAI_BASE_URL":
            value = super()._get_env("OPENAI_API_BASE")
        return value

    def build_cli_flags(self) -> str:
        extra = " ".join(f"-c {shlex.quote(f'{k}={_toml_value(v)}')}" for k, v in self._codex_config.items())
        base = super().build_cli_flags()
        return f"{base} {extra}".strip()

    async def install(self, environment: BaseEnvironment) -> None:
        result = await self.exec_as_agent(environment, command="codex --version", timeout_sec=120)
        if getattr(result, "return_code", 0):
            raise RuntimeError(f"preinstalled codex not usable from {_tools_bin()}:\n{getattr(result, 'stdout', '')}")

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        command = f'export PATH="{_tools_bin()}:$PATH"; {command}'
        return await super().exec_as_agent(
            environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec or self._exec_timeout_sec
        )
