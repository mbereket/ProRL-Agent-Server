"""Minimal Harbor agents for infrastructure tests (no model involved)."""

from __future__ import annotations

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class SleepAgent(BaseAgent):
    """Keeps its sandbox busy for ``sleep_sec`` (flush / teardown tests)."""

    def __init__(self, *args, sleep_sec: int = 600, **kwargs):
        super().__init__(*args, **kwargs)
        self._sleep_sec = int(sleep_sec)

    @staticmethod
    def name() -> str:
        return "sleep-test"

    def version(self) -> str | None:
        return "0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await environment.exec(f"sleep {self._sleep_sec}", timeout_sec=self._sleep_sec + 60)
