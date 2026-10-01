"""Engine refusals are kept on the session so the trainer can see why a harness stopped."""

from __future__ import annotations

from polar.gateway.storage import SessionStore


def test_upstream_error_is_kept_in_session_metadata() -> None:
    store = SessionStore()
    store.ensure_session("s1", None, None, None, task_id="t1", metadata={"group_id": 3})
    store.record_upstream_error("s1", "Upstream returned HTTP 400: The input (65625 tokens) is longer than the model's context length (65536 tokens).")
    store.record_upstream_error("s1", "x" * 5000)

    session = store.load_completion_session("s1")
    assert session.metadata["group_id"] == 3
    assert session.metadata["upstream_error"] == "x" * 2000
    assert session.metadata["upstream_error_count"] == 2
    assert store.get_session_metadata("s1")["metadata"]["upstream_error_count"] == 2
