from __future__ import annotations

import pytest

from polar.trajectory.evaluator.harbor import _reward_from_mapping


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"score": 3, "max_points": 7, "reward": 0.43}, 0.43),
        ({"reward": 1}, 1.0),
        ({"resolved": 0.5}, 0.5),
        ({"chart_selection": 1, "alt_text_insights": 0}, 0.5),
    ],
)
def test_reward_from_mapping(data, expected) -> None:
    assert _reward_from_mapping(data) == pytest.approx(expected)
