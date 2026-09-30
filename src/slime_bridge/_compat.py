"""Trainer-framework compatibility: Miles (radixark/miles) or Slime (THUDM/slime 0.3.0).

The bridge talks to the trainer through a handful of types -- ``Sample``, the
rollout output wrappers and the buffered data source -- that both frameworks
ship under their own package name. Miles is a Slime fork and keeps the same
shapes, with one semantic rename that matters here:

  * Slime 0.3.0 ``Sample.group_id``  ==  Miles ``Sample.rollout_id``: the loss
    aggregation unit. Every sample sharing the value is reduced as one unit
    (token-mean over the unit under sample-mean loss) and the DP/step schedule
    counts units, not samples. The bridge sets it to the trajectory (one agent
    session), so a session that fans out into several traces counts once.
  * Slime 0.3.0 also has ``Sample.session_id``; Miles does not. The session id
    always travels in ``sample.metadata["polar"]["session_id"]`` as well, and
    :func:`sample_session_id` reads it from there first.

Miles is preferred when both are importable (``POLAR_TRAINER_FRAMEWORK=slime``
forces Slime).
"""

from __future__ import annotations

import dataclasses
import inspect
import os
from functools import lru_cache
from typing import Any

_FRAMEWORKS = ("miles", "slime")


@lru_cache(maxsize=1)
def framework() -> str:
    """Name of the trainer framework in this process: ``miles`` or ``slime``."""
    forced = os.environ.get("POLAR_TRAINER_FRAMEWORK", "").strip().lower()
    candidates = (forced,) if forced else _FRAMEWORKS
    if forced and forced not in _FRAMEWORKS:
        raise ValueError(f"POLAR_TRAINER_FRAMEWORK must be one of {_FRAMEWORKS}, got {forced!r}")
    errors = []
    for name in candidates:
        try:
            __import__(f"{name}.utils.types")
            return name
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError(
        "Neither Miles nor Slime is importable; the bridge converts Polar rollouts into "
        "trainer samples and must run inside the trainer's environment. " + "; ".join(errors)
    )


def load_sample_type() -> Any:
    module = __import__(f"{framework()}.utils.types", fromlist=["Sample"])
    return module.Sample


def load_rollout_output_types() -> tuple[Any, Any]:
    """(RolloutFnTrainOutput, RolloutFnEvalOutput)."""
    module = __import__(f"{framework()}.rollout.base_types", fromlist=["RolloutFnTrainOutput"])
    return module.RolloutFnTrainOutput, module.RolloutFnEvalOutput


def load_buffered_data_source() -> Any:
    module = __import__(f"{framework()}.rollout.data_source", fromlist=["RolloutDataSourceWithBuffer"])
    return module.RolloutDataSourceWithBuffer


@lru_cache(maxsize=None)
def _sample_fields(sample_type: Any) -> frozenset[str]:
    if dataclasses.is_dataclass(sample_type):
        return frozenset(f.name for f in dataclasses.fields(sample_type))
    params = inspect.signature(sample_type).parameters
    return frozenset(name for name in params if name != "self")


def aggregation_field(sample_type: Any) -> str:
    """Name of the loss-aggregation-unit field on ``sample_type``."""
    fields = _sample_fields(sample_type)
    if "group_id" in fields:
        return "group_id"
    if "rollout_id" in fields:
        return "rollout_id"
    raise TypeError(
        f"{sample_type!r} has neither group_id (Slime 0.3.0) nor rollout_id (Miles): the trainer "
        "cannot aggregate the traces of one trajectory as one unit"
    )


def new_sample(sample_type: Any, *, unit_id: int, session_id: str | None = None, **fields: Any) -> Any:
    """Build a trainer Sample whose aggregation unit is ``unit_id``.

    ``session_id`` is set as a field only where the Sample type has one (Slime
    0.3.0); callers keep it in metadata for Miles.
    """
    known = _sample_fields(sample_type)
    fields[aggregation_field(sample_type)] = unit_id
    if session_id is not None and "session_id" in known:
        fields["session_id"] = session_id
    return sample_type(**fields)


def sample_unit_id(sample: Any) -> Any:
    """The sample's loss-aggregation unit (group_id / rollout_id), or None."""
    for name in ("group_id", "rollout_id"):
        value = getattr(sample, name, None)
        if value is not None:
            return value
    return None


def sample_session_id(sample: Any) -> Any:
    """Polar session id of a converted sample (metadata first, then the Slime field)."""
    metadata = getattr(sample, "metadata", None)
    if isinstance(metadata, dict):
        polar = metadata.get("polar")
        if isinstance(polar, dict) and polar.get("session_id"):
            return polar["session_id"]
    return getattr(sample, "session_id", None)
