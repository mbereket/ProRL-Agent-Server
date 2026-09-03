from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec
from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.models import Trajectory


class ExitRuntime(BaseRuntime):
    def __init__(self, tmp_path: Path, exit_code: int) -> None:
        super().__init__(RuntimeSpec(image="fake"), "session-1", tmp_path / "session")
        self.exit_code = exit_code

    @property
    def runtime_id(self) -> str:
        return "fake"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def exec(self, command: str, *, cwd=None, env=None, timeout_sec=None) -> ExecResult:
        if "test.sh" in command:
            return ExecResult(stdout="judge down", return_code=self.exit_code)
        if "reward.txt" in command:
            return ExecResult(stdout="0\n", return_code=0)
        return ExecResult(stdout="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None: ...

    async def upload_dir(self, local_path: str, remote_path: str) -> None: ...

    async def download_file(self, remote_path: str, local_path: str) -> None: ...

    async def download_dir(self, remote_path: str, local_path: str) -> None: ...


def _tests_dir(tmp_path: Path) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\n")
    return tests


def test_nonzero_exit_is_a_zero_reward_by_default(tmp_path: Path) -> None:
    evaluator = HarborEvaluator(tests_dir=str(_tests_dir(tmp_path)))
    result = asyncio.run(
        evaluator.evaluate(Trajectory(status="COMPLETED"), runtime=ExitRuntime(tmp_path, 2), artifacts_dir=tmp_path / "a")
    )
    assert result.outcome_reward == 0.0
    assert result.metadata["verifier_exit_code"] == 2


def test_fail_on_nonzero_exit_raises(tmp_path: Path) -> None:
    evaluator = HarborEvaluator(tests_dir=str(_tests_dir(tmp_path)), fail_on_nonzero_exit=True)
    with pytest.raises(RuntimeError, match="exited with 2"):
        asyncio.run(
            evaluator.evaluate(Trajectory(status="COMPLETED"), runtime=ExitRuntime(tmp_path, 2), artifacts_dir=tmp_path / "a")
        )
